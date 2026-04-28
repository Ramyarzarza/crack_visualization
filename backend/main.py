import io
import os
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Crack Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "Models"
SAMPLES_DIR = BASE_DIR / "samples"

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


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, nn.Module] = {}


def _load_model(model_name: str) -> nn.Module:
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_name}")

    checkpoint = torch.load(str(model_path), map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = UNetCompact(num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model = UNetCompact(num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint)
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
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


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
    model_name = Path(req.model).name

    sample_path = SAMPLES_DIR / sample_name
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_name}")

    # Load model (cached after first load)
    model = _load_model(model_name)

    # Load image
    img_gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise HTTPException(status_code=500, detail="Could not read image file")

    # Run inference
    if req.split:
        grid = req.split_grid
        mask = _predict_split(model, img_gray, req.resolution, grid, req.confidence_threshold)
    else:
        mask = _predict_whole(model, img_gray, req.resolution, req.confidence_threshold)

    # Apply class filter — zero out classes the user doesn't want to see
    if "crack" not in req.show_classes:
        mask[mask == LINE_CLASS] = 0
    if "shape" not in req.show_classes:
        mask[mask == SHAPE_CLASS] = 0

    # Build visualizations
    overlay_bgr = _create_overlay(img_gray, mask, alpha=0.4)
    mask_rgb = _mask_to_color_rgb(mask)

    # Statistics
    total = mask.size
    line_pct = float((mask == LINE_CLASS).sum() / total * 100)
    shape_pct = float((mask == SHAPE_CLASS).sum() / total * 100)
    defect_pct = line_pct + shape_pct

    return {
        "original": _ndarray_to_base64(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), is_bgr=True),
        "mask": _ndarray_to_base64(mask_rgb, is_bgr=False),
        "overlay": _ndarray_to_base64(overlay_bgr, is_bgr=True),
        "stats": {
            "line_percentage": round(line_pct, 3),
            "shape_percentage": round(shape_pct, 3),
            "defect_percentage": round(defect_pct, 3),
            "has_crack": line_pct > 0.1,
            "has_shape": shape_pct > 0.1,
            "image_size": {"width": int(img_gray.shape[1]), "height": int(img_gray.shape[0])},
        },
    }
