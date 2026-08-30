"""Load a trained checkpoint and run it on a full-resolution image.

Same aspect-preserving letterbox the training used; the padding is cropped off
the probability map before it's scaled back to the original size.
"""

import cv2
import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD
from .model import build_model


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["encoder"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, ckpt["imgsz"], ckpt.get("metrics", {})


@torch.no_grad()
def predict_prob(model, imgsz: int, device: torch.device, rgb: np.ndarray) -> np.ndarray:
    """Per-pixel text probability at the image's original resolution."""
    h, w = rgb.shape[:2]
    s = imgsz / max(h, w)
    nh, nw = round(h * s), round(w * s)
    canvas = np.zeros((imgsz, imgsz, 3), np.uint8)
    oy, ox = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = cv2.resize(
        rgb, (nw, nh), interpolation=cv2.INTER_AREA
    )
    x = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    prob = model(x).sigmoid()[0, 0].cpu().numpy()[oy : oy + nh, ox : ox + nw]
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def mask_from_prob(prob: np.ndarray, threshold: float = 0.5, dilate: int = 0) -> np.ndarray:
    mask = (prob > threshold).astype(np.uint8) * 255
    if dilate > 0:
        k = 2 * dilate + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask
