"""Inpaint the labelled text out of comic pages (LaMa, via the localizer).

Input: a `.cvlabel` where each page carries text shapes (polygons/boxes) to
remove and region boxes to keep. Output: a new `.cvlabel` with cleaned images
and the region annotations, ready for `dataset.generate_from_backgrounds`.

Label naming: both conventions are accepted --
    remove:  Text | Text In Bubble | Text on Page | Text On Page
    keep:    Speech Bubble | Bubble | Free Draw Area
"""

import asyncio

import cv2
import numpy as np
import torch
from tqdm import tqdm

from . import cvlabel

REMOVE = ("Text", "Text In Bubble", "Text on Page", "Text On Page")
KEEP = ("Speech Bubble", "Bubble", "Free Draw Area")

DEFAULT_CLEANER_REPO = "TareHimself/AnimeMangaInpainting-torchscript"
DEFAULT_CLEANER_FILE = "anime_manga_lama.pt"


def _resolve_cleaner_model(cleaner_model: str) -> str:
    if cleaner_model:
        return cleaner_model
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DEFAULT_CLEANER_REPO, DEFAULT_CLEANER_FILE)


async def _clean(
    labels: str,
    out: str,
    cleaner_model: str = "",
    device: str = "cuda:0",
    pad: int = 6,
    remove: tuple = REMOVE,
    keep: tuple = KEEP,
) -> None:
    from comic_localizer.cleaning.lama import LamaCleaner
    from comic_localizer.core.constants import DetectionType, SegmentationType
    from comic_localizer.core.plugin import DetectionResult, SegmentationResult

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable -> CPU (slow).")
        device = "cpu"
    cleaner = LamaCleaner(
        model_path=_resolve_cleaner_model(cleaner_model), device=torch.device(device)
    )

    out_pages: list[cvlabel.Sample] = []
    for page in tqdm(cvlabel.read(labels), desc="pages"):
        h, w = page.image.shape[:2]
        shapes = page.by_label(*remove)
        kept = page.by_label(*keep)
        if not shapes:
            out_pages.append(cvlabel.Sample(page.name, page.image, kept, page.split))
            continue

        mask = np.zeros((h, w), np.uint8)
        segments, detections = [], []
        for a in shapes:
            poly = a.polygon()
            cv2.fillPoly(mask, [poly], 255)
            segments.append(SegmentationResult(SegmentationType.Text, poly, 1.0))
            x1, y1, x2, y2 = a.xyxy
            bbox = np.array(
                [max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)],
                np.int32,
            )
            detections.append(DetectionResult(DetectionType.TextInBubble, bbox, 1.0))

        cleaned = (await cleaner.clean([page.image], [mask], [segments], [detections]))[0]
        out_pages.append(cvlabel.Sample(page.name, cleaned, kept, page.split))

    cvlabel.write(out, out_pages)
    regions = sum(len(p.annotations) for p in out_pages)
    print(f"{len(out_pages)} pages, {regions} draw regions -> {out}")


def clean_pages(labels: str, out: str, cleaner_model: str = "", **kw) -> None:
    asyncio.run(_clean(labels, out, cleaner_model, **kw))
