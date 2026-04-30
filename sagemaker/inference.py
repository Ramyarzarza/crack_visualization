"""
SageMaker inference script for crack detection.
SageMaker calls: model_fn, input_fn, predict_fn, output_fn
"""
import io, json, os, sys, types as _types, base64
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

NUM_CLASSES  = 3
LINE_INDEX   = 1
SHAPE_INDEX  = 2
LINE_CLASS   = 255
SHAPE_CLASS  = 125


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


# Pickle compatibility: checkpoint was saved in Jupyter __main__
for _n in ("__main__",):
    _m = sys.modules.get(_n) or _types.ModuleType(_n)
    sys.modules[_n] = _m
    _m.UNetCompact = UNetCompact
    _m.ConvBlock   = ConvBlock


def model_fn(model_dir: str):
    """Load all .pt files from model_dir and return a dict {filename: model}."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for pt_path in Path(model_dir).glob("*.pt"):
        ckpt = torch.load(str(pt_path), map_location=device, weights_only=False)
        if isinstance(ckpt, nn.Module):
            model = ckpt
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model = UNetCompact(); model.load_state_dict(ckpt["model_state_dict"])
        else:
            model = UNetCompact(); model.load_state_dict(ckpt)
        models[pt_path.name] = model.to(device).eval()
        print(f"  Loaded model: {pt_path.name}")
    if not models:
        raise RuntimeError(f"No .pt files found in {model_dir}")
    return models


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


def predict_fn(data: dict, models: dict):
    img, res  = data["image"], data["resolution"]
    thr, sc   = data["confidence_threshold"], data["show_classes"]
    oh, ow    = img.shape

    # Select model by name; fall back to first available
    model_name = data.get("model_name", "")
    if model_name and model_name in models:
        model = models[model_name]
    else:
        model = next(iter(models.values()))

    device = next(model.parameters()).device

    t = torch.from_numpy(cv2.resize(img, (res, res))).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0).to(device)

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
