"""Box maths and letterbox resize."""

import cv2
import numpy as np

Box = tuple[int, int, int, int]


def round_to_8(v: int) -> int:
    return v - (v % 8)


def clamp_box(xyxy, w: int, h: int, min_side: int = 8) -> Box | None:
    x1, y1, x2, y2 = xyxy
    box = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
    if box[2] - box[0] < min_side or box[3] - box[1] < min_side:
        return None
    return box


def boxes_disjoint(a: Box, b: Box, gap: int = 0) -> bool:
    return (
        a[2] + gap <= b[0]
        or b[2] + gap <= a[0]
        or a[3] + gap <= b[1]
        or b[3] + gap <= a[1]
    )


def random_boxes(
    width: int,
    height: int,
    count: int,
    rand,
    *,
    min_size: int = 80,
    gap: int = 8,
) -> list[Box]:
    """Up to `count` non-overlapping `(x1, y1, x2, y2)` boxes inside the
    surface, each at least `min_size` per side with `gap` px between them."""
    max_w, max_h = width - 2 * gap, height - 2 * gap
    if min_size > max_w or min_size > max_h:
        return []

    boxes: list[Box] = []
    for _ in range(count * 10):
        if len(boxes) == count:
            break
        bw = rand.randint(min_size, max_w)
        bh = rand.randint(min_size, max_h)
        x1 = rand.randint(gap, width - gap - bw)
        y1 = rand.randint(gap, height - gap - bh)
        box = (x1, y1, x1 + bw, y1 + bh)
        if all(boxes_disjoint(box, other, gap) for other in boxes):
            boxes.append(box)
    return boxes


def letterbox(
    img: np.ndarray, mask: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resize keeping aspect (long side -> `size`), centre-pad to `size` square.
    Pad is 0 for both -- an unambiguous 'no text' region for the mask."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out_i = np.zeros((size, size, 3), img.dtype)
    out_m = np.zeros((size, size), mask.dtype)
    y, x = (size - nh) // 2, (size - nw) // 2
    out_i[y : y + nh, x : x + nw] = img
    out_m[y : y + nh, x : x + nw] = mask
    return out_i, out_m
