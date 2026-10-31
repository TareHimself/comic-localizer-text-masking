"""Cut a composed page into per-region crops for training.

The segmenter always runs on a detector crop, never a whole page, so training
on crops matches deployment and sidesteps the aspect-ratio squish of full
webtoon pages. Each text box becomes a crop, grown by a random pad to mimic a
loose detector box; a few text-free crops are added as negatives.
"""

from collections.abc import Iterator

import cv2
import numpy as np

from .render import Box, Sample


def _degrade(img: np.ndarray, rand, prob: float = 0.7) -> np.ndarray:
    """JPEG ringing + a touch of blur / noise, so a synthetic crop's edges
    aren't crisper than a real scan's."""
    if rand.random() >= prob:
        return img
    if rand.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rand.uniform(0.4, 0.9))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rand.randint(55, 92)])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rand.random() < 0.4:
        rng = np.random.default_rng(rand.getrandbits(32))
        noise = rng.normal(0.0, rand.uniform(1.5, 5.0), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def page_crops(
    page: Sample,
    boxes: list[Box],
    rand,
    *,
    pad: tuple[float, float] = (0.08, 0.45),
    tight_pad: tuple[float, float] = (-0.15, 0.05),
    tight_prob: float = 0.25,
    min_size: int = 40,
    negative_prob: float = 0.35,
    max_negatives: int = 2,
    degrade_prob: float = 0.7,
) -> Iterator[Sample]:
    h, w = page.image.shape[:2]

    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        # a quarter of the time crop tight (or slightly into) the text box, so
        # the model sees text that runs off the frame edge, not just centred
        lo, hi = tight_pad if rand.random() < tight_prob else pad
        cx1 = max(0, int(x1 - bw * rand.uniform(lo, hi)))
        cy1 = max(0, int(y1 - bh * rand.uniform(lo, hi)))
        cx2 = min(w, int(x2 + bw * rand.uniform(lo, hi)))
        cy2 = min(h, int(y2 + bh * rand.uniform(lo, hi)))
        if cx2 - cx1 >= min_size and cy2 - cy1 >= min_size:
            yield Sample(
                _degrade(page.image[cy1:cy2, cx1:cx2].copy(), rand, degrade_prob),
                page.mask[cy1:cy2, cx1:cx2].copy(),
            )

    for _ in range(max_negatives):
        if rand.random() >= negative_prob:
            continue
        cw = rand.randint(min_size, max(min_size, w // 2))
        ch = rand.randint(min_size, max(min_size, h // 2))
        if cw > w or ch > h:
            continue
        cx1, cy1 = rand.randint(0, w - cw), rand.randint(0, h - ch)
        m = page.mask[cy1 : cy1 + ch, cx1 : cx1 + cw]
        if m.mean() < 2:  # essentially no text
            yield Sample(
                _degrade(page.image[cy1 : cy1 + ch, cx1 : cx1 + cw].copy(), rand, degrade_prob),
                m.copy(),
            )
