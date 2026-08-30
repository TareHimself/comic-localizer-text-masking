"""`MaskDataset` -- loads from raw crop dirs or `pack` memmap packs.

A root is a pack for a split if `<root>/<split>/meta.json` exists (memmap slice
+ unpack); otherwise it decodes + letterboxes the PNGs. `--data` can mix both;
each image->mask path is resolved individually so `<shard>_<n>.png` crop names
never collide between datasets.
"""

import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..geometry import letterbox

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img: np.ndarray) -> torch.Tensor:
    img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """CHW normalized tensor -> HWC uint8 RGB, for logging."""
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip((arr * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255).astype(np.uint8)


def _zoom(
    img: np.ndarray, mask: np.ndarray, p: float = 0.4, min_scale: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a random sub-window and scale it back to full size -- up to ~3x
    magnification. Teaches scale invariance so tightly-cropped large text (SFX,
    titles) isn't out of distribution. The window is biased to sit over the
    text, but may clip it -- partial glyphs are still valid mask targets."""
    if random.random() > p:
        return img, mask
    h, w = mask.shape
    s = random.uniform(min_scale, 1.0)
    cw, ch = max(8, int(w * s)), max(8, int(h * s))
    ys, xs = np.where(mask > 127)
    if len(xs):  # keep the text roughly centred in the window
        cx, cy = (int(xs.min()) + int(xs.max())) // 2, (int(ys.min()) + int(ys.max())) // 2
        x0 = min(max(random.randint(cx - cw + 1, cx), 0), w - cw)
        y0 = min(max(random.randint(cy - ch + 1, cy), 0), h - ch)
    else:
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
    img = cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(
        mask[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST
    )
    return img, mask


def _augment(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img, mask = _zoom(img, mask)
    if random.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if random.random() < 0.2:
        img, mask = img[::-1], mask[::-1]
    if random.random() < 0.5:
        img = np.clip(img * (0.6 + random.random() * 0.8), 0, 255)
    return img, mask


def _raw_pairs(root: str, split: str) -> list[tuple[str, str]]:
    out = []
    for img in sorted(glob.glob(os.path.join(root, split, "images", "*.png"))):
        mask = os.path.join(
            os.path.dirname(os.path.dirname(img)), "masks", os.path.basename(img)
        )
        out.append((img, mask))
    return out


class MaskDataset(Dataset):
    def __init__(
        self, roots: list[str], split: str, imgsz: int, train: bool, limit: int = 0
    ):
        self.imgsz = imgsz
        self.train = train
        self._packs: dict = {}  # (root, split) -> (images_mm, masks_mm, S) per process
        self.entries: list = []
        for r in roots:
            meta_path = os.path.join(r, split, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta["imgsz"] != imgsz:
                    raise ValueError(
                        f"{r} packed at imgsz {meta['imgsz']}, --imgsz is {imgsz} "
                        "-- re-run tm-pack"
                    )
                self.entries += [("pack", r, split, i) for i in range(meta["n"])]
            else:
                self.entries += [("raw", *p) for p in _raw_pairs(r, split)]
        if limit:
            self.entries = self.entries[:limit]
        if not self.entries:
            raise FileNotFoundError(f"no {split} samples under any of {roots}")

    def _pack(self, root: str, split: str):
        key = (root, split)
        if key not in self._packs:
            with open(os.path.join(root, split, "meta.json")) as f:
                m = json.load(f)
            n, s, mb = m["n"], m["imgsz"], m["mask_bytes"]
            self._packs[key] = (
                np.memmap(os.path.join(root, split, "images.u8"), np.uint8, "r",
                          shape=(n, s, s, 3)),
                np.memmap(os.path.join(root, split, "masks.u8"), np.uint8, "r",
                          shape=(n, mb)),
                s,
            )
        return self._packs[key]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        if entry[0] == "pack":
            _, root, split, i = entry
            imgs, masks, s = self._pack(root, split)
            img = np.array(imgs[i])
            mask = (
                np.unpackbits(np.array(masks[i]))[: s * s].reshape(s, s).astype(np.uint8)
                * 255
            )
        else:
            _, img_path, mask_path = entry
            img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
            img, mask = letterbox(img, mask, self.imgsz)

        if self.train:
            img, mask = _augment(img.astype(np.float32), mask)

        target = torch.from_numpy(
            np.ascontiguousarray((mask > 127).astype(np.float32))
        ).unsqueeze(0)
        return normalize(img), target


def loader(ds: Dataset, batch: int, workers: int, train: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=workers > 0,
    )
