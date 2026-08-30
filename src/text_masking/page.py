"""Operations on labelled `.cvlabel` pages: flip, derive draw regions.

The generator reads `Speech Bubble` / `Bubble` and `Free Draw Area`
annotations off a cleaned page and turns them into draw boxes for synthetic
text. `bubble_text_box` uses the localizer's `compute_draw_bbox` (largest
rectangle inside the bubble outline); `free_text_boxes` scatters small boxes
inside a canvas region, sometimes none (a labelled spot with no text -- a
useful hard negative).
"""

import cv2

from . import cvlabel
from .geometry import Box, clamp_box, random_boxes

BUBBLE_LABELS = ("Speech Bubble", "Bubble")
AREA_LABELS = ("Free Draw Area",)
FLIPS = (None, 1, 0, -1)  # cv2.flip codes: None=as-is, 1=h, 0=v, -1=both


def flip_page(page: cvlabel.Sample, code: int) -> cvlabel.Sample:
    """A page flipped by cv2 `code`, annotations transformed to match."""
    h, w = page.image.shape[:2]
    fx, fy = code in (1, -1), code in (0, -1)
    anns = [
        cvlabel.Annotation(
            a.label,
            a.kind,
            [((w - 1 - x) if fx else x, (h - 1 - y) if fy else y) for x, y in a.points],
        )
        for a in page.annotations
    ]
    return cvlabel.Sample(page.name, cv2.flip(page.image, code), anns, page.split)


def bubble_text_box(page_image, ann: cvlabel.Annotation, w: int, h: int) -> Box | None:
    """Where text goes inside a labelled bubble on a cleaned page."""
    from comic_localizer.utils import compute_draw_bbox

    box = clamp_box(ann.xyxy, w, h)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    dx1, dy1, dx2, dy2 = (int(v) for v in compute_draw_bbox(page_image[y1:y2, x1:x2]))
    if dx2 - dx1 >= 8 and dy2 - dy1 >= 8:
        return clamp_box((x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2), w, h)
    return box


def free_text_boxes(
    area: Box, rand, fill_prob: float = 0.65, max_bits: int = 3
) -> list[Box]:
    """0-`max_bits` sub-boxes inside `area` for independent bits of free text.
    Empty with probability `1 - fill_prob`."""
    if rand.random() >= fill_prob:
        return []
    ax1, ay1, ax2, ay2 = area
    aw, ah = ax2 - ax1, ay2 - ay1
    inner = random_boxes(
        aw, ah, rand.randint(1, max_bits), rand,
        min_size=max(16, min(44, aw - 12, ah - 12)), gap=6,
    )
    return [(ax1 + x1, ay1 + y1, ax1 + x2, ay1 + y2) for x1, y1, x2, y2 in inner]
