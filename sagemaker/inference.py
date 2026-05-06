"""
SageMaker inference script for crack detection.
SageMaker calls: model_fn, input_fn, predict_fn, output_fn

NOTE: This file must be self-contained — SageMaker cannot import from the
workspace. Both UNetCompact and UNet (from U_net.py) are inlined here.
"""
import io, json, os, sys, types as _types, base64
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from PIL import Image

NUM_CLASSES  = 3
LINE_INDEX   = 1
SHAPE_INDEX  = 2
LINE_CLASS   = 255
SHAPE_CLASS  = 125


# ---------------------------------------------------------------------------
# Architecture 1: UNetCompact  (Generalized_dataset_* models)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
    def forward(self, x): return self.block(x)


class UNetCompact(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1=ConvBlock(1,16);   self.pool1=nn.MaxPool2d(2)
        self.enc2=ConvBlock(16,32);  self.pool2=nn.MaxPool2d(2)
        self.enc3=ConvBlock(32,64);  self.pool3=nn.MaxPool2d(2)
        self.enc4=ConvBlock(64,128); self.pool4=nn.MaxPool2d(2)
        self.bottleneck=ConvBlock(128,256)
        self.up4=nn.ConvTranspose2d(256,128,2,2); self.dec4=ConvBlock(256,128)
        self.up3=nn.ConvTranspose2d(128,64,2,2);  self.dec3=ConvBlock(128,64)
        self.up2=nn.ConvTranspose2d(64,32,2,2);   self.dec2=ConvBlock(64,32)
        self.up1=nn.ConvTranspose2d(32,16,2,2);   self.dec1=ConvBlock(32,16)
        self.outc=nn.Conv2d(16,num_classes,1)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool1(e1))
        e3=self.enc3(self.pool2(e2)); e4=self.enc4(self.pool3(e3))
        b=self.bottleneck(self.pool4(e4))
        d4=self.dec4(torch.cat([self.up4(b),e4],1))
        d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.outc(d1)


# ---------------------------------------------------------------------------
# Architecture 2: UNet  (U-net_model_* models — inlined from U_net.py)
# ---------------------------------------------------------------------------
def _conv3x3(in_ch, out_ch, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=padding, bias=bias, groups=groups)

def _upconv2x2(in_ch, out_ch, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                         nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1))

def _conv1x1(in_ch, out_ch, groups=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        self.pooling = pooling
        self.conv1 = _conv3x3(in_channels, out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels)
        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super().__init__()
        self.merge_mode = merge_mode
        self.upconv = _upconv2x2(in_channels, out_channels, mode=up_mode)
        if merge_mode == 'concat':
            self.conv1 = _conv3x3(2 * out_channels, out_channels)
        else:
            self.conv1 = _conv3x3(out_channels, out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1) if self.merge_mode == 'concat' else from_up + from_down
        return F.relu(self.conv2(F.relu(self.conv1(x))))


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose', merge_mode='concat'):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        down_convs, up_convs = [], []
        outs = None
        for i in range(depth):
            ins  = in_channels if i == 0 else outs
            outs = start_filts * (2 ** i)
            down_convs.append(DownConv(ins, outs, pooling=(i < depth - 1)))

        for i in range(depth - 1):
            ins  = outs
            outs = ins // 2
            up_convs.append(UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode))

        self.down_convs  = nn.ModuleList(down_convs)
        self.up_convs    = nn.ModuleList(up_convs)
        self.conv_final  = _conv1x1(outs, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_outs = []
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            x = module(encoder_outs[-(i + 2)], x)
        return self.conv_final(x)


# ---------------------------------------------------------------------------
# Architecture auto-detection
# ---------------------------------------------------------------------------
def _build_model_from_state_dict(state_dict: dict) -> nn.Module:
    if any(k.startswith("down_convs.") for k in state_dict):
        num_classes = state_dict["conv_final.weight"].shape[0]
        in_channels = state_dict["down_convs.0.conv1.weight"].shape[1]
        depth       = sum(1 for k in state_dict if k.startswith("down_convs.") and k.endswith(".conv1.weight"))
        start_filts = state_dict["down_convs.0.conv1.weight"].shape[0]
        return UNet(num_classes=num_classes, in_channels=in_channels,
                    depth=depth, start_filts=start_filts)
    return UNetCompact(num_classes=NUM_CLASSES)


# ---------------------------------------------------------------------------
# Pickle compatibility: checkpoint was saved in Jupyter __main__
# ---------------------------------------------------------------------------
for _n in ("__main__", "__mp_main__"):
    _m = sys.modules.get(_n) or _types.ModuleType(_n)
    sys.modules[_n] = _m
    _m.UNetCompact = UNetCompact
    _m.ConvBlock   = ConvBlock
    _m.UNet        = UNet
    _m.DownConv    = DownConv
    _m.UpConv      = UpConv


def _load_single_model(pt_path: Path) -> nn.Module:
    """Load one .pt file and return an eval-mode model on CPU."""
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):
        model = ckpt.module if isinstance(ckpt, nn.DataParallel) else ckpt
    elif isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model = _build_model_from_state_dict(state_dict)
        model.load_state_dict(state_dict)
    else:
        raise RuntimeError(f"Unsupported checkpoint type for {pt_path.name}: {type(ckpt)}")
    return model.eval()


def model_fn(model_dir: str):
    """Return a lazy loader dict: {filename: Path}.  Models are loaded on first use."""
    pt_files = list(Path(model_dir).glob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {model_dir}")
    model_paths = {p.name: p for p in pt_files}
    print(f"  Found {len(model_paths)} model(s) — will load on first use: {list(model_paths.keys())}")
    # Return a lazy-loading wrapper instead of pre-loaded models
    return {"_paths": model_paths, "_cache": {}}


def input_fn(request_body: bytes, content_type: str):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    p = json.loads(request_body)
    arr = np.frombuffer(base64.b64decode(p["image"]), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    return {
        "image":                img,
        "model_name":           p.get("model_name", ""),
        "resolution":           int(p.get("resolution", 800)),
        "confidence_threshold": float(p.get("confidence_threshold", 0.5)),
        "show_classes":         p.get("show_classes", ["crack", "shape"]),
    }


def predict_fn(data: dict, model_store: dict):
    img, res  = data["image"], data["resolution"]
    thr, sc   = data["confidence_threshold"], data["show_classes"]
    oh, ow    = img.shape

    # Resolve the requested model from the lazy store
    model_name = data.get("model_name", "")
    paths: dict = model_store["_paths"]
    cache: dict = model_store["_cache"]

    # Pick the target filename
    if model_name and model_name in paths:
        target = model_name
    else:
        target = next(iter(paths))

    # Load & cache on first use
    if target not in cache:
        print(f"  Lazy-loading model: {target}")
        cache[target] = _load_single_model(paths[target])
        print(f"  Model loaded: {target}")

    model  = cache[target]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    t = torch.from_numpy(cv2.resize(img, (res, res))).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    in_channels = getattr(model, 'in_channels', 1)
    if in_channels > 1:
        t = t.expand(-1, in_channels, -1, -1)  # [1, C, H, W]
    t = t.to(device)

    with torch.no_grad():
        logits = model(t)

    probs = F.softmax(logits, 1)
    mp, pi = probs.max(1)
    pi = pi.squeeze(0).cpu().numpy().astype(np.uint8)
    mp = mp.squeeze(0).cpu().numpy()

    mask = np.zeros_like(pi, dtype=np.uint8)
    c    = mp >= thr
    mask[(pi == LINE_INDEX)  & c] = LINE_CLASS
    mask[(pi == SHAPE_INDEX) & c] = SHAPE_CLASS
    mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

    if "crack" not in sc: mask[mask == LINE_CLASS]  = 0
    if "shape" not in sc: mask[mask == SHAPE_CLASS] = 0
    return mask


def output_fn(mask: np.ndarray, accept: str):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == SHAPE_CLASS] = [255, 215,   0]
    color[mask == LINE_CLASS]  = [255,   0,   0]
    buf = io.BytesIO()
    Image.fromarray(color).save(buf, "PNG")
    lp = float((mask == LINE_CLASS).sum()  / mask.size * 100)
    sp = float((mask == SHAPE_CLASS).sum() / mask.size * 100)
    return json.dumps({
        "mask":              base64.b64encode(buf.getvalue()).decode(),
        "line_percentage":   round(lp, 3),
        "shape_percentage":  round(sp, 3),
        "defect_percentage": round(lp + sp, 3),
        "has_crack":  lp > 0.1,
        "has_shape":  sp > 0.1,
    }), "application/json"
