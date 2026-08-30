"""Cut a composed page into per-region crops for training.

The segmenter always runs on a detector crop, never a whole page, so training
on crops matches deployment and sidesteps the aspect-ratio squish of full
webtoon pages. Each text box becomes a crop, grown by a random pad to mimic a
loose detector box; a few text-free crops are added as negatives.
"""

from collections.abc import Iterator

from .render import Box, Sample


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
                page.image[cy1:cy2, cx1:cx2].copy(),
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
            yield Sample(page.image[cy1 : cy1 + ch, cx1 : cx1 + cw].copy(), m.copy())
