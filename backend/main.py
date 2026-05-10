import io
import os
import base64
import json
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image as PILImage
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Crack Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return all unhandled exceptions as JSON so the frontend can parse them."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )

# ---------------------------------------------------------------------------
# SageMaker mode (optional)
# Set SAGEMAKER_ENDPOINT env var to route inference to a SageMaker endpoint
# instead of running PyTorch locally.
# Set USE_LOCAL_INFERENCE = True to force local PyTorch regardless of SAGEMAKER_ENDPOINT.
# ---------------------------------------------------------------------------
USE_LOCAL_INFERENCE  = False  # ← change to True to force local PyTorch

_SAGEMAKER_ENDPOINT  = os.environ.get("SAGEMAKER_ENDPOINT", "").strip()
_SAGEMAKER_REGION    = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

if USE_LOCAL_INFERENCE:
    _SAGEMAKER_ENDPOINT = ""  # override: ignore any SAGEMAKER_ENDPOINT
    _sm_runtime = None
    print("Local inference mode (USE_LOCAL_INFERENCE=true)")
elif _SAGEMAKER_ENDPOINT:
    try:
        import boto3 as _boto3
        from botocore.config import Config as _BotocoreConfig
        _sm_runtime = _boto3.client(
            "sagemaker-runtime",
            region_name=_SAGEMAKER_REGION,
            config=_BotocoreConfig(read_timeout=120, retries={"max_attempts": 0}),
        )
        print(f"SageMaker mode enabled — endpoint: {_SAGEMAKER_ENDPOINT}")
    except ImportError:
        raise RuntimeError("boto3 is required for SageMaker mode: pip install boto3")
else:
    _sm_runtime = None
    print("Local inference mode (no SAGEMAKER_ENDPOINT set)")


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent

# Make the workspace root importable so U_net.py can be imported
import sys as _sys_path_setup
if str(BASE_DIR) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(BASE_DIR))
from U_net import UNet, DownConv, UpConv
MODELS_DIR    = BASE_DIR / "Models"
SAMPLES_DIR   = BASE_DIR / "samples"
LABELING_DIR  = BASE_DIR / "Labeling"
LABEL_IMAGES_DIR = LABELING_DIR / "Images"
LABEL_MASKS_DIR  = LABELING_DIR / "Masks"

# Ensure labeling directories exist at startup
LABEL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABEL_MASKS_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND_INDEX = 0
LINE_INDEX = 1
SHAPE_INDEX = 2
NUM_CLASSES = 3

LINE_CLASS = 255
SHAPE_CLASS = 125

ALLOWED_RESOLUTIONS = {256, 512, 800, 1600}
ALLOWED_GRIDS = {2, 3, 4}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model architecture (identical to notebook)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetCompact(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)

        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.outc(d1)


# ---------------------------------------------------------------------------
# Pickle compatibility: register model classes under every module name that
# PyTorch / pickle might have used when the checkpoint was saved (Jupyter
# saves under '__main__'; uvicorn --reload uses '__mp_main__').
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types
for _mod_name in ("__main__", "__mp_main__"):
    _mod = _sys.modules.get(_mod_name)
    if _mod is None:
        _mod = _types.ModuleType(_mod_name)
        _sys.modules[_mod_name] = _mod
    _mod.UNetCompact = UNetCompact
    _mod.ConvBlock = ConvBlock
    _mod.UNet = UNet
    _mod.DownConv = DownConv
    _mod.UpConv = UpConv


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, nn.Module] = {}

# ---------------------------------------------------------------------------
# Filter cache — stores the raw (unfiltered) mask + image from the last /predict
# so that /filter can re-apply post-processing without re-running inference.
# ---------------------------------------------------------------------------
_filter_cache: dict = {}


def _build_model_from_state_dict(state_dict: dict) -> nn.Module:
    """Instantiate the correct architecture by inspecting state dict keys."""
    if any(k.startswith("down_convs.") for k in state_dict):
        # UNet architecture (U_net.py)
        num_classes  = state_dict["conv_final.weight"].shape[0]
        in_channels  = state_dict["down_convs.0.conv1.weight"].shape[1]
        depth        = sum(1 for k in state_dict if k.startswith("down_convs.") and k.endswith(".conv1.weight"))
        start_filts  = state_dict["down_convs.0.conv1.weight"].shape[0]
        return UNet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts)
    else:
        # UNetCompact architecture
        return UNetCompact(num_classes=NUM_CLASSES)


def _load_model(model_name: str) -> nn.Module:
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_name}")

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

    if isinstance(checkpoint, nn.Module):
        # Unwrap DataParallel — it hard-codes device_ids=[0] (cuda) which
        # causes a crash when running on CPU or a different device.
        model = checkpoint.module if isinstance(checkpoint, nn.DataParallel) else checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Strip "module." prefix produced by DataParallel saves
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model = _build_model_from_state_dict(state_dict)
        model.load_state_dict(state_dict)
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported checkpoint type: {type(checkpoint)}")

    model = model.to(DEVICE)
    model.eval()
    _model_cache[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------
def _decode_prediction(pred_tensor: torch.Tensor, confidence_threshold: float = 0.0) -> np.ndarray:
    """Convert model logits to a mask.

    If confidence_threshold > 0, a pixel is only labelled as crack/shape when
    the softmax probability of the winning class exceeds the threshold;
    otherwise it falls back to background.  threshold=0 reproduces the
    original argmax-only behaviour.
    """
    probs = F.softmax(pred_tensor, dim=1)          # [1, C, H, W]  in [0, 1]
    max_probs, pred_indices = probs.max(dim=1)     # each [1, H, W]
    pred_indices = pred_indices.squeeze(0).cpu().numpy().astype(np.uint8)
    max_probs    = max_probs.squeeze(0).cpu().numpy()

    mask = np.zeros_like(pred_indices, dtype=np.uint8)  # everything → background
    confident = max_probs >= confidence_threshold        # boolean map
    mask[(pred_indices == LINE_INDEX)  & confident] = LINE_CLASS
    mask[(pred_indices == SHAPE_INDEX) & confident] = SHAPE_CLASS
    return mask


def _predict_tile(
    model: nn.Module,
    tile_gray: np.ndarray,
    resolution: int,
    confidence_threshold: float = 0.0,
) -> np.ndarray:
    """Run inference on a single grayscale tile; resize output back to tile dims."""
    original_h, original_w = tile_gray.shape
    resized = cv2.resize(tile_gray, (resolution, resolution))
    tensor = torch.from_numpy(resized).float() / 255.0
    # Expand to the number of channels the model expects (usually 1, sometimes 3)
    in_channels = getattr(model, 'in_channels', 1)
    tensor = tensor.unsqueeze(0).unsqueeze(0)           # [1, 1, H, W]
    if in_channels > 1:
        tensor = tensor.expand(-1, in_channels, -1, -1) # [1, C, H, W]
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        prediction = model(tensor)

    mask = _decode_prediction(prediction, confidence_threshold)
    return cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)


def _predict_whole(
    model: nn.Module,
    img_gray: np.ndarray,
    resolution: int,
    confidence_threshold: float = 0.0,
) -> np.ndarray:
    return _predict_tile(model, img_gray, resolution, confidence_threshold)


def _predict_split(
    model: nn.Module,
    img_gray: np.ndarray,
    resolution: int,
    grid: int,
    confidence_threshold: float = 0.0,
) -> np.ndarray:
    """Split into grid×grid tiles, predict each, stitch masks together."""
    H, W = img_gray.shape
    tile_h = H // grid
    tile_w = W // grid
    full_mask = np.zeros((H, W), dtype=np.uint8)

    for row in range(grid):
        for col in range(grid):
            y1 = row * tile_h
            y2 = y1 + tile_h if row < grid - 1 else H
            x1 = col * tile_w
            x2 = x1 + tile_w if col < grid - 1 else W
            tile = img_gray[y1:y2, x1:x2]
            full_mask[y1:y2, x1:x2] = _predict_tile(model, tile, resolution, confidence_threshold)

    return full_mask


def _apply_intensity_filter(
    mask: np.ndarray,
    img_gray: np.ndarray,
    intensity_min: int = 0,
    intensity_max: int = 255,
) -> np.ndarray:
    """Zero out mask pixels whose underlying grayscale intensity is outside [min, max]."""
    if intensity_min <= 0 and intensity_max >= 255:
        return mask  # fast path
    result = mask.copy()
    out_of_range = (img_gray < intensity_min) | (img_gray > intensity_max)
    result[out_of_range] = 0
    return result


def _postprocess_crack_mask(
    mask: np.ndarray,
    close_gap_size: int = 0,
    min_crack_area: int = 0,
    max_circularity: float = 1.0,
) -> np.ndarray:
    """Filter crack (LINE_CLASS) pixels to suppress noise and keep linear structures.

    Parameters
    ----------
    close_gap_size:
        Radius of the elliptical structuring element for morphological closing
        (dilate then erode).  Bridges gaps between nearby crack segments so
        that close-together spots merge into one connected component before
        the area/circularity filters run.  0 = disabled.
    min_crack_area:
        Discard connected components (8-connectivity) whose pixel area is
        below this value.  0 = disabled.
    max_circularity:
        Discard components whose circularity (4π·area/perimeter²) exceeds this
        value.  A perfect circle = 1.0; any elongated line or crack network
        (including cross/plus shapes) scores far below 1.0.  1.0 = disabled.
    """
    if close_gap_size <= 0 and min_crack_area <= 0 and max_circularity >= 1.0:
        return mask  # fast path – nothing to do

    crack_binary = (mask == LINE_CLASS).astype(np.uint8)

    # 1. Gap closing — dilate to find nearby segments, then fill the bridge.
    #    Any dilated component that contains at least one original crack pixel
    #    is kept in full (including the bridging pixels), connecting the gaps.
    if close_gap_size > 0:
        k = close_gap_size * 2 + 1  # radius → kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(crack_binary, kernel)
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        merged = np.zeros_like(crack_binary)
        for lid in range(1, num_labels):
            comp_pixels = labels == lid
            if (crack_binary[comp_pixels] > 0).any():  # component touches a real crack pixel
                merged[comp_pixels] = 1  # fill the whole component, including bridge pixels
        crack_binary = merged

    # 2. Per-component area + circularity filtering
    if min_crack_area > 0 or max_circularity < 1.0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            crack_binary, connectivity=8
        )
        keep = np.zeros_like(crack_binary)
        for label_id in range(1, num_labels):  # 0 = background
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            # Area gate — CC_STAT_AREA counts actual crack pixels, never fills holes
            if min_crack_area > 0 and area < min_crack_area:
                continue
            # Circularity gate: 4π·area/perimeter²
            #   circle ≈ 1.0  |  thin line or crack network << 1.0
            #   Cross/plus shapes also score low because their perimeter is large
            #   relative to enclosed area — they are correctly preserved.
            if max_circularity < 1.0:
                comp_mask = (labels == label_id).astype(np.uint8)
                cnts, _ = cv2.findContours(
                    comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if cnts:
                    perimeter = sum(cv2.arcLength(c, True) for c in cnts)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > max_circularity:
                            continue  # too blob-like → discard
            keep[labels == label_id] = 1  # copy only the original pixels
        crack_binary = keep

    result = mask.copy()
    result[mask == LINE_CLASS] = 0          # clear original crack pixels
    result[crack_binary == 1] = LINE_CLASS  # restore only the kept ones
    return result


def _detect_sample_circle(
    img_gray: np.ndarray,
) -> Optional[Tuple[int, int, int]]:
    """Detect the dominant circular sample boundary using Hough circles.

    Returns (cx, cy, radius) in original image pixels, or None if not found.
    """
    h, w = img_gray.shape
    min_dim = min(h, w)
    # Blur strongly to suppress texture noise; only the bold circle edge survives
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dim // 2,           # expect at most one dominant circle
        param1=80,                       # Canny high threshold
        param2=35,                       # accumulator threshold (lower = more detections)
        minRadius=min_dim // 5,          # ignore tiny circles
        maxRadius=int(min_dim * 0.65),   # ignore circles larger than the image
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    # Return the largest detected circle
    best = circles[np.argmax(circles[:, 2])]
    return int(best[0]), int(best[1]), int(best[2])


def _apply_circle_mask(
    mask: np.ndarray,
    circle: tuple[int, int, int],
    margin: int = 0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    """Zero out mask pixels that fall outside the detected circle.

    margin > 0 expands the circle, margin < 0 shrinks it.
    offset_x/y shift the circle centre in pixels.
    """
    cx, cy, r = circle
    r_adj = max(1, r + margin)
    circle_region = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(circle_region, (cx + offset_x, cy + offset_y), r_adj, 1, thickness=-1)
    result = mask.copy()
    result[circle_region == 0] = 0
    return result


def _create_overlay(img_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    overlay[mask == SHAPE_CLASS] = [0, 215, 255]  # gold (BGR)
    overlay[mask == LINE_CLASS] = [0, 0, 255]      # red  (BGR)
    return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)


def _mask_to_color_rgb(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == SHAPE_CLASS] = [255, 215, 0]  # gold
    color[mask == LINE_CLASS] = [255, 0, 0]      # red
    return color


def _ndarray_to_base64(img: np.ndarray, is_bgr: bool = True) -> str:
    """Convert numpy image array to PNG base64 string."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if is_bgr else img
    pil_img = PILImage.fromarray(img_rgb)
    # Limit max dimension to 1600 px for display performance
    max_dim = 1600
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    model: str
    sample: str
    resolution: int
    split: bool
    split_grid: int = 2
    confidence_threshold: float = 0.5
    show_classes: list[str] = ["crack", "shape"]
    close_gap_size: int = 0
    min_crack_area: int = 0
    max_circularity: float = 1.0
    circle_mask: bool = False
    circle_mask_margin: int = 0
    circle_mask_offset_x: int = 0
    circle_mask_offset_y: int = 0
    intensity_min: int = 0
    intensity_max: int = 255

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: int) -> int:
        if v not in ALLOWED_RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}")
        return v

    @field_validator("split_grid")
    @classmethod
    def validate_grid(cls, v: int) -> int:
        if v not in ALLOWED_GRIDS:
            raise ValueError(f"split_grid must be one of {sorted(ALLOWED_GRIDS)}")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("show_classes")
    @classmethod
    def validate_show_classes(cls, v: list[str]) -> list[str]:
        valid = {"crack", "shape"}
        for item in v:
            if item not in valid:
                raise ValueError(f"show_classes items must be one of {valid}")
        return v

    @field_validator("close_gap_size")
    @classmethod
    def validate_close_gap_size(cls, v: int) -> int:
        if not 0 <= v <= 15:
            raise ValueError("close_gap_size must be 0–15")
        return v

    @field_validator("min_crack_area")
    @classmethod
    def validate_min_crack_area(cls, v: int) -> int:
        if not 0 <= v <= 5000:
            raise ValueError("min_crack_area must be 0–5000")
        return v

    @field_validator("max_circularity")
    @classmethod
    def validate_max_circularity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("max_circularity must be 0.0–1.0")
        return v

    @field_validator("circle_mask_margin")
    @classmethod
    def validate_circle_mask_margin(cls, v: int) -> int:
        if not -200 <= v <= 200:
            raise ValueError("circle_mask_margin must be -200–200")
        return v

    @field_validator("circle_mask_offset_x")
    @classmethod
    def validate_circle_mask_offset_x(cls, v: int) -> int:
        if not -500 <= v <= 500:
            raise ValueError("circle_mask_offset_x must be -500–500")
        return v

    @field_validator("circle_mask_offset_y")
    @classmethod
    def validate_circle_mask_offset_y(cls, v: int) -> int:
        if not -500 <= v <= 500:
            raise ValueError("circle_mask_offset_y must be -500–500")
        return v

    @field_validator("intensity_min")
    @classmethod
    def validate_intensity_min(cls, v: int) -> int:
        if not 0 <= v <= 255:
            raise ValueError("intensity_min must be 0–255")
        return v

    @field_validator("intensity_max")
    @classmethod
    def validate_intensity_max(cls, v: int) -> int:
        if not 0 <= v <= 255:
            raise ValueError("intensity_max must be 0–255")
        return v


# ---------------------------------------------------------------------------
# Filter request/response models
# ---------------------------------------------------------------------------
class FilterRequest(BaseModel):
    close_gap_size: int = 0
    min_crack_area: int = 0
    max_circularity: float = 1.0
    circle_mask: bool = False
    circle_mask_margin: int = 0
    circle_mask_offset_x: int = 0
    circle_mask_offset_y: int = 0
    intensity_min: int = 0
    intensity_max: int = 255

    @field_validator("close_gap_size")
    @classmethod
    def validate_close_gap_size(cls, v: int) -> int:
        if not 0 <= v <= 15:
            raise ValueError("close_gap_size must be 0–15")
        return v

    @field_validator("min_crack_area")
    @classmethod
    def validate_min_crack_area(cls, v: int) -> int:
        if not 0 <= v <= 5000:
            raise ValueError("min_crack_area must be 0–5000")
        return v

    @field_validator("max_circularity")
    @classmethod
    def validate_max_circularity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("max_circularity must be 0.0–1.0")
        return v

    @field_validator("circle_mask_margin")
    @classmethod
    def validate_circle_mask_margin(cls, v: int) -> int:
        if not -200 <= v <= 200:
            raise ValueError("circle_mask_margin must be -200–200")
        return v

    @field_validator("circle_mask_offset_x")
    @classmethod
    def validate_circle_mask_offset_x(cls, v: int) -> int:
        if not -500 <= v <= 500:
            raise ValueError("circle_mask_offset_x must be -500–500")
        return v

    @field_validator("circle_mask_offset_y")
    @classmethod
    def validate_circle_mask_offset_y(cls, v: int) -> int:
        if not -500 <= v <= 500:
            raise ValueError("circle_mask_offset_y must be -500–500")
        return v

    @field_validator("intensity_min")
    @classmethod
    def validate_intensity_min(cls, v: int) -> int:
        if not 0 <= v <= 255:
            raise ValueError("intensity_min must be 0–255")
        return v

    @field_validator("intensity_max")
    @classmethod
    def validate_intensity_max(cls, v: int) -> int:
        if not 0 <= v <= 255:
            raise ValueError("intensity_max must be 0–255")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    mode = "sagemaker" if _SAGEMAKER_ENDPOINT else "local"
    return {"status": "ok", "device": str(DEVICE), "inference_mode": mode}


@app.post("/filter")
def apply_filter(req: FilterRequest):
    """Re-apply post-processing to the cached raw mask from the last /predict call.
    Returns updated mask, overlay and stats without re-running model inference.
    """
    if "raw_mask" not in _filter_cache:
        raise HTTPException(
            status_code=400,
            detail="No prediction cached. Run /predict first.",
        )

    img_gray = _filter_cache["img_gray"]
    mask = _filter_cache["raw_mask"].copy()

    # 1. Intensity filter — first step: remove predictions on pixels outside the brightness range
    mask = _apply_intensity_filter(mask, img_gray, req.intensity_min, req.intensity_max)

    # 2. Circle mask
    if req.circle_mask:
        circle = _filter_cache.get("circle")
        if circle is not None:
            mask = _apply_circle_mask(mask, circle, req.circle_mask_margin, req.circle_mask_offset_x, req.circle_mask_offset_y)

    mask = _postprocess_crack_mask(
        mask, req.close_gap_size, req.min_crack_area, req.max_circularity
    )

    overlay_bgr = _create_overlay(img_gray, mask, alpha=0.4)
    mask_rgb    = _mask_to_color_rgb(mask)

    total     = mask.size
    line_pct  = float((mask == LINE_CLASS).sum()  / total * 100)
    shape_pct = float((mask == SHAPE_CLASS).sum() / total * 100)

    raw_circle = _filter_cache.get("circle")
    circle_out = (
        {"cx": int(raw_circle[0] + req.circle_mask_offset_x), "cy": int(raw_circle[1] + req.circle_mask_offset_y), "radius": int(raw_circle[2] + req.circle_mask_margin)}
        if raw_circle is not None else None
    )

    return {
        "mask":    _ndarray_to_base64(mask_rgb, is_bgr=False),
        "overlay": _ndarray_to_base64(overlay_bgr, is_bgr=True),
        "circle":  circle_out,
        "stats": {
            "line_percentage":   round(line_pct,  3),
            "shape_percentage":  round(shape_pct, 3),
            "defect_percentage": round(line_pct + shape_pct, 3),
            "has_crack":  line_pct  > 0.1,
            "has_shape":  shape_pct > 0.1,
            "image_size": {
                "width":  int(img_gray.shape[1]),
                "height": int(img_gray.shape[0]),
            },
        },
    }


@app.get("/models")
def list_models():
    models = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")])
    return {"models": models}


@app.get("/samples")
def list_samples():
    samples = sorted(
        f for f in os.listdir(SAMPLES_DIR)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    )
    return {"samples": samples}


@app.post("/predict")
def predict(req: PredictRequest):
    # Validate paths (no directory traversal)
    sample_name = Path(req.sample).name
    model_name  = Path(req.model).name

    sample_path = SAMPLES_DIR / sample_name
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_name}")

    # Load image (always done locally — it's just file I/O)
    img_gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise HTTPException(status_code=500, detail="Could not read image file")

    # ── Inference ────────────────────────────────────────────────────────────
    if _SAGEMAKER_ENDPOINT:
        # ── SageMaker path ──
        _, img_encoded = cv2.imencode(".png", img_gray)
        img_b64 = base64.b64encode(img_encoded.tobytes()).decode()
        payload = json.dumps({
            "image":                img_b64,
            "model_name":           model_name,
            "resolution":           req.resolution,
            "confidence_threshold": req.confidence_threshold,
            "show_classes":         req.show_classes,
        })
        try:
            response  = _sm_runtime.invoke_endpoint(
                EndpointName=_SAGEMAKER_ENDPOINT,
                ContentType="application/json",
                Body=payload,
            )
        except Exception as _e:
            _ename = type(_e).__name__
            print(f"SageMaker invoke_endpoint error [{_ename}]: {_e}")
            if "ReadTimeout" in _ename or "Timeout" in _ename:
                raise HTTPException(
                    status_code=504,
                    detail="SageMaker inference timed out. This model is too heavy for the serverless endpoint. Try a smaller model or lower resolution.",
                )
            raise HTTPException(status_code=502, detail=f"SageMaker error ({_ename}): {_e}")
        sm_result = json.loads(response["Body"].read())

        # Decode mask from SageMaker for overlay generation
        mask_bytes = base64.b64decode(sm_result["mask"])
        mask_pil   = PILImage.open(io.BytesIO(mask_bytes)).convert("RGB")
        mask_rgb   = np.array(mask_pil)

        # Rebuild grayscale mask for overlay
        mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        mask[np.all(mask_rgb == [255, 0, 0],   axis=2)] = LINE_CLASS
        mask[np.all(mask_rgb == [255, 215, 0], axis=2)] = SHAPE_CLASS

        overlay_bgr = _create_overlay(img_gray, mask, alpha=0.4)

        return {
            "original": _ndarray_to_base64(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), is_bgr=True),
            "mask":     _ndarray_to_base64(mask_rgb, is_bgr=False),
            "overlay":  _ndarray_to_base64(overlay_bgr, is_bgr=True),
            "stats": {
                "line_percentage":   sm_result["line_percentage"],
                "shape_percentage":  sm_result["shape_percentage"],
                "defect_percentage": sm_result["defect_percentage"],
                "has_crack":  sm_result["has_crack"],
                "has_shape":  sm_result["has_shape"],
                "image_size": {"width": int(img_gray.shape[1]), "height": int(img_gray.shape[0])},
            },
        }

    else:
        # ── Local PyTorch path ──
        model = _load_model(model_name)

        if req.split:
            mask = _predict_split(model, img_gray, req.resolution, req.split_grid, req.confidence_threshold)
        else:
            mask = _predict_whole(model, img_gray, req.resolution, req.confidence_threshold)

        # Cache the raw mask (before any post-processing) so /filter can reuse it
        _filter_cache["img_gray"] = img_gray
        _filter_cache["raw_mask"] = mask.copy()
        # Detect and cache the sample circle (used by circle-mask feature)
        _filter_cache["circle"] = _detect_sample_circle(img_gray)

        # 1. Intensity filter — first step
        mask = _apply_intensity_filter(mask, img_gray, req.intensity_min, req.intensity_max)

        # 2. Circle mask
        if req.circle_mask:
            circle = _filter_cache.get("circle")
            if circle is not None:
                mask = _apply_circle_mask(mask, circle, req.circle_mask_margin, req.circle_mask_offset_x, req.circle_mask_offset_y)

        # Post-process crack channel
        mask = _postprocess_crack_mask(
            mask, req.close_gap_size, req.min_crack_area, req.max_circularity
        )

        # Apply class filter
        if "crack" not in req.show_classes:
            mask[mask == LINE_CLASS]  = 0
        if "shape" not in req.show_classes:
            mask[mask == SHAPE_CLASS] = 0

        overlay_bgr = _create_overlay(img_gray, mask, alpha=0.4)
        mask_rgb    = _mask_to_color_rgb(mask)

        total     = mask.size
        line_pct  = float((mask == LINE_CLASS).sum()  / total * 100)
        shape_pct = float((mask == SHAPE_CLASS).sum() / total * 100)

        raw_circle = _filter_cache.get("circle")
        circle_out = (
            {"cx": int(raw_circle[0] + req.circle_mask_offset_x), "cy": int(raw_circle[1] + req.circle_mask_offset_y), "radius": int(raw_circle[2] + req.circle_mask_margin)}
            if raw_circle is not None else None
        )

        return {
            "original": _ndarray_to_base64(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), is_bgr=True),
            "mask":     _ndarray_to_base64(mask_rgb, is_bgr=False),
            "overlay":  _ndarray_to_base64(overlay_bgr, is_bgr=True),
            "circle":   circle_out,
            "stats": {
                "line_percentage":   round(line_pct,  3),
                "shape_percentage":  round(shape_pct, 3),
                "defect_percentage": round(line_pct + shape_pct, 3),
                "has_crack":  line_pct  > 0.1,
                "has_shape":  shape_pct > 0.1,
                "image_size": {"width": int(img_gray.shape[1]), "height": int(img_gray.shape[0])},
            },
        }


# ---------------------------------------------------------------------------
# Labeling endpoints
# ---------------------------------------------------------------------------

class SaveMaskRequest(BaseModel):
    filename: str               # must match a file in Labeling/Images
    mask: str                   # base64-encoded PNG — raw mask (no filters), used for reloading
    mask_filtered: Optional[str] = None  # base64-encoded PNG — post-processed mask (with filters applied)


class DenoiseRequest(BaseModel):
    mask: str       # base64-encoded grayscale PNG
    min_area: int = 100

    @field_validator("min_area")
    @classmethod
    def validate_min_area(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("min_area must be 0–100")
        return v


@app.post("/labeling/denoise")
def labeling_denoise(req: DenoiseRequest):
    """Remove connected components smaller than min_area from a label mask."""
    try:
        img_bytes = base64.b64decode(req.mask)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 mask data")

    arr  = np.frombuffer(img_bytes, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=400, detail="Could not decode mask image")

    if req.min_area > 0:
        result = mask.copy()
        for class_value in (LINE_CLASS, SHAPE_CLASS):
            binary = (mask == class_value).astype(np.uint8)
            if not binary.any():
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            keep = np.zeros_like(binary)
            for lid in range(1, num_labels):
                if int(stats[lid, cv2.CC_STAT_AREA]) >= req.min_area:
                    keep[labels == lid] = 1
            result[mask == class_value] = 0
            result[keep == 1] = class_value
        mask = result

    success, buf = cv2.imencode(".png", mask)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode mask")
    return {"mask": base64.b64encode(buf.tobytes()).decode("utf-8")}


# ---------------------------------------------------------------------------
# Evaluation helpers + endpoint
# ---------------------------------------------------------------------------

# Per-image comparison thumbnails from the last /evaluate run
_eval_image_cache: dict[str, dict] = {}


def _create_comparison_overlay(
    img_gray: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Composite overlay showing TP/FP/FN for cracks only (shapes ignored).

    Colour coding:
      Green  — True Positive  (both pred and GT say crack)
      Red    — False Positive (pred says crack, GT says background)
      Cyan   — False Negative (GT says crack, pred says background)

    If gt_mask is None, only prediction cracks are shown in red.
    """
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    pred_crack = pred_mask == LINE_CLASS

    if gt_mask is None:
        overlay[pred_crack] = [0, 0, 255]          # red  — no GT available
    else:
        gt_crack = gt_mask == 255
        overlay[pred_crack & gt_crack]   = [0, 255, 0]   # green  TP
        overlay[pred_crack & ~gt_crack]  = [0, 0, 255]   # red    FP
        overlay[~pred_crack & gt_crack]  = [255, 255, 0] # cyan   FN

    return cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)


def _thumb_b64(img_bgr: np.ndarray, max_dim: int = 900) -> str:
    """Resize to max_dim and return base64-encoded PNG."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(img_rgb)
    w, h = pil.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _compute_crack_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Binary segmentation metrics for the crack class only.

    pred_mask : model output where LINE_CLASS (255) = crack
    gt_mask   : grayscale ground-truth where 255 = crack, 0 = background
    """
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    pred_crack = pred_mask == LINE_CLASS
    gt_crack   = gt_mask   == 255

    tp = int(( pred_crack &  gt_crack).sum())
    fp = int(( pred_crack & ~gt_crack).sum())
    fn = int((~pred_crack &  gt_crack).sum())

    # Perfect negative case: no cracks anywhere, none predicted
    if tp == 0 and fp == 0 and fn == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}

    dice      = round(2 * tp / (2 * tp + fp + fn), 4)
    iou       = round(tp / (tp + fp + fn), 4)
    precision = round(tp / (tp + fp), 4) if tp + fp > 0 else 0.0
    recall    = round(tp / (tp + fn), 4) if tp + fn > 0 else 0.0
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


class EvaluateRequest(BaseModel):
    model: str
    resolution: int
    split: bool = False
    split_grid: int = 2
    confidence_threshold: float = 0.5
    close_gap_size: int = 0
    min_crack_area: int = 0
    max_circularity: float = 1.0
    circle_mask: bool = False
    circle_mask_margin: int = 0
    circle_mask_offset_x: int = 0
    circle_mask_offset_y: int = 0
    intensity_min: int = 0
    intensity_max: int = 255

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: int) -> int:
        if v not in ALLOWED_RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}")
        return v

    @field_validator("split_grid")
    @classmethod
    def validate_grid(cls, v: int) -> int:
        if v not in ALLOWED_GRIDS:
            raise ValueError(f"split_grid must be one of {sorted(ALLOWED_GRIDS)}")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("close_gap_size")
    @classmethod
    def validate_close_gap_size(cls, v: int) -> int:
        if not 0 <= v <= 15:
            raise ValueError("close_gap_size must be 0–15")
        return v

    @field_validator("min_crack_area")
    @classmethod
    def validate_min_crack_area(cls, v: int) -> int:
        if not 0 <= v <= 5000:
            raise ValueError("min_crack_area must be 0–5000")
        return v

    @field_validator("max_circularity")
    @classmethod
    def validate_max_circularity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("max_circularity must be 0.0–1.0")
        return v


@app.post("/evaluate")
def evaluate_test_set(req: EvaluateRequest):
    """Run inference on every labeled test image and return crack segmentation metrics.

    For each image that has a ground-truth mask in Labeling/Masks, two predictions are computed:
    - raw_pred    : model output before any post-processing
    - filtered_pred: model output after applying the current post-processing settings

    Each prediction is compared against both ground-truth variants:
    - raw_gt      : Labeling/Masks/<stem>_mask.png
    - filtered_gt : Labeling/Masks/<stem>_mask_filtered.png

    Metrics returned (crack class only): Dice, IoU, Precision, Recall.
    """
    model_name = Path(req.model).name
    model = _load_model(model_name)

    per_image: list[dict] = []
    _eval_image_cache.clear()

    for img_path in sorted(LABEL_IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = img_path.stem                               # e.g. "front_0200"
        raw_gt_path      = LABEL_MASKS_DIR / f"{stem}_mask.png"
        filtered_gt_path = LABEL_MASKS_DIR / f"{stem}_mask_filtered.png"

        if not raw_gt_path.exists() and not filtered_gt_path.exists():
            continue  # no ground-truth for this image → skip

        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        # ── Raw prediction (model output only, no post-processing) ──────────
        if req.split:
            raw_pred = _predict_split(
                model, img_gray, req.resolution, req.split_grid, req.confidence_threshold
            )
        else:
            raw_pred = _predict_whole(
                model, img_gray, req.resolution, req.confidence_threshold
            )

        # ── Filtered prediction (apply current post-processing settings) ────
        filt_pred = raw_pred.copy()
        filt_pred = _apply_intensity_filter(filt_pred, img_gray, req.intensity_min, req.intensity_max)
        circle = _detect_sample_circle(img_gray) if req.circle_mask else None
        if req.circle_mask and circle is not None:
            filt_pred = _apply_circle_mask(
                filt_pred, circle,
                req.circle_mask_margin, req.circle_mask_offset_x, req.circle_mask_offset_y,
            )
        filt_pred = _postprocess_crack_mask(
            filt_pred, req.close_gap_size, req.min_crack_area, req.max_circularity
        )

        entry: dict = {"image": img_path.name}
        gts: dict[str, np.ndarray] = {}  # gt_key → circle-masked GT array

        for gt_key, gt_path in [("raw_gt", raw_gt_path), ("filtered_gt", filtered_gt_path)]:
            if not gt_path.exists():
                entry[gt_key] = None
                continue
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                entry[gt_key] = None
                continue
            # Apply circle mask to GT as well so both sides are clipped identically
            if req.circle_mask and circle is not None:
                gt = _apply_circle_mask(
                    gt, circle,
                    req.circle_mask_margin, req.circle_mask_offset_x, req.circle_mask_offset_y,
                )
            gts[gt_key] = gt
            entry[gt_key] = {
                "raw_pred":      _compute_crack_metrics(raw_pred,  gt),
                "filtered_pred": _compute_crack_metrics(filt_pred, gt),
            }

        per_image.append(entry)

        # Cache combined comparison thumbnails — one per (pred_variant × gt_variant)
        _eval_image_cache[img_path.name] = {
            f"{pred_k}_vs_{gt_k}": _thumb_b64(
                _create_comparison_overlay(img_gray, pm, gts.get(gt_k))
            )
            for pred_k, pm in [("raw_pred", raw_pred), ("filtered_pred", filt_pred)]
            for gt_k in ("raw_gt", "filtered_gt")
        }

    # ── Aggregate (mean across images) for every pred × gt combination ──────
    aggregate: dict = {}
    for pred_k in ("raw_pred", "filtered_pred"):
        for gt_k in ("raw_gt", "filtered_gt"):
            vals = [
                r[gt_k][pred_k]
                for r in per_image
                if r.get(gt_k) is not None
            ]
            if vals:
                aggregate[f"{pred_k}_vs_{gt_k}"] = {
                    m: round(sum(v[m] for v in vals) / len(vals), 4)
                    for m in ("dice", "iou", "precision", "recall")
                }

    return {
        "results":     per_image,
        "aggregate":   aggregate,
        "n_evaluated": len(per_image),
    }


@app.get("/evaluate/compare/{filename}")
def evaluate_compare(filename: str):
    """Return cached comparison thumbnails for a single image from the last /evaluate run.

    Returns:
        original      — grayscale original image
        raw_pred      — red overlay: unfiltered model prediction
        filtered_pred — red overlay: post-processed prediction
        raw_gt        — cyan overlay: original ground-truth labels (or null)
        filtered_gt   — cyan overlay: filtered ground-truth labels (or null)
    """
    safe = Path(filename).name
    if safe not in _eval_image_cache:
        raise HTTPException(
            status_code=404,
            detail="No comparison data cached. Run /evaluate first.",
        )
    return _eval_image_cache[safe]


@app.get("/labeling/images")
def labeling_list_images():
    """List images available for labeling in Labeling/Images."""
    images = sorted(
        f for f in os.listdir(LABEL_IMAGES_DIR)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    )
    return {"images": images}


@app.get("/labeling/image/{filename}")
def labeling_get_image(filename: str):
    """Return a labeling image as base64 PNG at its original resolution (no downscaling)."""
    safe_name = Path(filename).name
    img_path = LABEL_IMAGES_DIR / safe_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {safe_name}")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read image")
    h, w = img.shape
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode image")
    encoded = base64.b64encode(buf.tobytes()).decode("utf-8")
    return {"image": encoded, "width": w, "height": h}


@app.get("/labeling/mask/{filename}")
def labeling_get_mask(filename: str):
    """Return the existing mask for an image, or null if none exists."""
    safe_name = Path(filename).name
    mask_path = LABEL_MASKS_DIR / f"{Path(safe_name).stem}_mask.png"
    if not mask_path.exists():
        return {"mask": None}
    with open(mask_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return {"mask": encoded}


@app.post("/labeling/save")
def labeling_save_mask(req: SaveMaskRequest):
    """Save raw and filtered mask PNGs to Labeling/Masks."""
    safe_name = Path(req.filename).name
    if not (LABEL_IMAGES_DIR / safe_name).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {safe_name}")

    stem = Path(safe_name).stem

    # Save raw mask (used for reloading)
    raw_path = LABEL_MASKS_DIR / f"{stem}_mask.png"
    try:
        raw_bytes = base64.b64decode(req.mask)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 mask data")
    with open(raw_path, "wb") as f:
        f.write(raw_bytes)

    # Save filtered mask if provided
    saved = [raw_path.name]
    if req.mask_filtered is not None:
        filtered_path = LABEL_MASKS_DIR / f"{stem}_mask_filtered.png"
        try:
            filtered_bytes = base64.b64decode(req.mask_filtered)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 mask_filtered data")
        with open(filtered_path, "wb") as f:
            f.write(filtered_bytes)
        saved.append(filtered_path.name)

    return {"saved": saved}


@app.get("/labeling/download-zip")
def labeling_download_zip():
    """Stream the entire Labeling folder (Images + Masks) as a zip archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in (LABEL_IMAGES_DIR, LABEL_MASKS_DIR):
            if folder.exists():
                for fpath in sorted(folder.iterdir()):
                    if fpath.is_file():
                        arcname = Path("Labeling") / folder.name / fpath.name
                        zf.write(fpath, arcname)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="Labeling.zip"'},
    )
