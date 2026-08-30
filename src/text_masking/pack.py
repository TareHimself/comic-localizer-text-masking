"""Pre-render a crop dataset into memory-mapped packs for fast training.

For every (image, mask) pair under each root's `train/ valid/ (test/)`, it
letterboxes to `imgsz` square and appends to a memmap. Per split:

    <out>/<split>/images.u8   memmap (N, S, S, 3) uint8
    <out>/<split>/masks.u8    memmap (N, mask_bytes) uint8  (bit-packed)
    <out>/<split>/meta.json   {n, imgsz, mask_bytes, sources}

`sources` carries provenance -- per contributing root, its crop count and the
`manifest.json` tm-generate wrote (background .cvlabel, repeats, seed, ...).
tm-train surfaces this in the run config / wandb.

Source datasets are untouched. `tm-train --data <out>` loads straight from the
memmaps. Re-run when imgsz changes or data is added.
"""

import glob
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

from .geometry import letterbox


def _source_entries(roots: list[str], split: str) -> list[dict]:
    """Provenance for one split: each contributing root's crop count and the
    `manifest.json` tm-generate left behind (if any)."""
    entries = []
    for r in roots:
        p = pairs(r, split)
        if not p:
            continue
        entry = {"root": r, "n": len(p)}
        mpath = os.path.join(r, split, "manifest.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                entry["manifest"] = json.load(f)
        entries.append(entry)
    return entries


def pairs(root: str, split: str) -> list[tuple[str, str]]:
    out = []
    for ip in sorted(glob.glob(os.path.join(root, split, "images", "*.png"))):
        mp = os.path.join(
            os.path.dirname(os.path.dirname(ip)), "masks", os.path.basename(ip)
        )
        out.append((ip, mp))
    return out


@dataclass
class _Shard:
    images_path: str
    masks_path: str
    imgsz: int
    mask_bytes: int
    n: int
    lo: int
    hi: int
    pairs: list


def _fill(sh: _Shard) -> int:
    imgs = np.memmap(sh.images_path, np.uint8, "r+", shape=(sh.n, sh.imgsz, sh.imgsz, 3))
    masks = np.memmap(sh.masks_path, np.uint8, "r+", shape=(sh.n, sh.mask_bytes))
    for i in range(sh.lo, sh.hi):
        ip, mp = sh.pairs[i]
        img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros(img.shape[:2], np.uint8)
        img, m = letterbox(img, m, sh.imgsz)
        imgs[i] = img
        masks[i] = np.packbits((m > 127).reshape(-1))
    imgs.flush()
    masks.flush()
    return sh.hi - sh.lo


def pack(roots: list[str], out: str, imgsz: int, workers: int = 0) -> None:
    s = imgsz
    mask_bytes = math.ceil(s * s / 8)
    workers = workers or min(os.cpu_count() or 4, 8)

    splits = sorted(
        {
            sp
            for r in roots
            for sp in ("train", "valid", "test")
            if os.path.isdir(os.path.join(r, sp, "images"))
        }
    )
    for split in splits:
        all_pairs = [p for r in roots for p in pairs(r, split)]
        n = len(all_pairs)
        d = os.path.join(out, split)
        os.makedirs(d, exist_ok=True)
        ip, mp = os.path.join(d, "images.u8"), os.path.join(d, "masks.u8")
        np.memmap(ip, np.uint8, "w+", shape=(n, s, s, 3)).flush()
        np.memmap(mp, np.uint8, "w+", shape=(n, mask_bytes)).flush()

        e = [round(n * k / workers) for k in range(workers + 1)]
        shards = [
            _Shard(ip, mp, s, mask_bytes, n, e[k], e[k + 1], all_pairs)
            for k in range(workers)
            if e[k + 1] > e[k]
        ]
        with ProcessPoolExecutor(max_workers=len(shards)) as ex, tqdm(
            total=n, desc=split
        ) as bar:
            for done in ex.map(_fill, shards):
                bar.update(done)

        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(
                {
                    "n": n,
                    "imgsz": s,
                    "mask_bytes": mask_bytes,
                    "sources": _source_entries(roots, split),
                },
                f,
                indent=2,
            )

    print(f"packed {splits} -> {out}")
