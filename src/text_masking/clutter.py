"""Procedural non-text clutter for synthetic backgrounds.

Drawn behind the text so the segmenter has to learn *text vs. structured
non-text* (strokes, hatching, screentone, panel borders) rather than
"sharp edge = text". Never enters the mask -- `compose` only masks the glyphs.
"""

import cv2
import numpy as np


def _gray(rand, lo=60, hi=180) -> tuple[int, int, int]:
    v = rand.randint(lo, hi)
    return (v, v, v)


def _region(img: np.ndarray, rand):
    """A random contiguous sub-rectangle: (x, y, tile). Write the tile back."""
    h, w = img.shape[:2]
    ph = rand.randint(h // 6, h)
    pw = rand.randint(w // 6, w)
    y = rand.randint(0, h - ph)
    x = rand.randint(0, w - pw)
    return x, y, np.ascontiguousarray(img[y : y + ph, x : x + pw])


# --- whole-image elements (surface is contiguous) ----------------------------


def strokes(img: np.ndarray, rand) -> None:
    h, w = img.shape[:2]
    for _ in range(rand.randint(4, 24)):
        pts = np.array(
            [(rand.randint(0, w), rand.randint(0, h)) for _ in range(rand.randint(2, 5))],
            np.int32,
        )
        cv2.polylines(img, [pts], False, _gray(rand), rand.randint(1, 3), cv2.LINE_AA)


def shapes(img: np.ndarray, rand) -> None:
    h, w = img.shape[:2]
    for _ in range(rand.randint(1, 6)):
        c = (rand.randint(0, w), rand.randint(0, h))
        color = _gray(rand, 30, 200)
        thick = rand.choice([-1, 1, 2, 3])
        roll = rand.random()
        if roll < 0.4:
            cv2.circle(img, c, rand.randint(8, max(9, min(w, h) // 3)), color, thick, cv2.LINE_AA)
        elif roll < 0.7:
            axes = (rand.randint(10, max(11, w // 3)), rand.randint(10, max(11, h // 3)))
            cv2.ellipse(img, c, axes, rand.randint(0, 180), 0, 360, color, thick, cv2.LINE_AA)
        else:
            pts = np.array(
                [(rand.randint(0, w), rand.randint(0, h)) for _ in range(rand.randint(3, 6))],
                np.int32,
            )
            cv2.polylines(img, [pts], True, color, max(1, thick), cv2.LINE_AA)


def borders(img: np.ndarray, rand) -> None:
    h, w = img.shape[:2]
    color = _gray(rand, 0, 40)
    thick = rand.randint(2, 6)
    for _ in range(rand.randint(1, 3)):
        if rand.random() < 0.5:
            x = rand.randint(0, w)
            cv2.line(img, (x, 0), (x, h), color, thick)
        else:
            y = rand.randint(0, h)
            cv2.line(img, (0, y), (w, y), color, thick)


def gradient(img: np.ndarray, rand) -> None:
    h, w = img.shape[:2]
    vertical = rand.random() < 0.5
    ramp = np.linspace(rand.uniform(-45, 45), rand.uniform(-45, 45), h if vertical else w)
    delta = ramp[:, None, None] if vertical else ramp[None, :, None]
    img[:] = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


# --- region elements (drawn on a tile, written back) -------------------------


def hatching(tile: np.ndarray, rand) -> None:
    ph, pw = tile.shape[:2]
    gap = rand.randint(4, 14)
    thick = rand.randint(1, 2)
    color = _gray(rand, 60, 170)
    slope = rand.uniform(-2, 2)
    for c in range(-pw, pw + ph, gap):
        cv2.line(tile, (c, 0), (int(c + slope * ph), ph), color, thick, cv2.LINE_AA)
    if rand.random() < 0.4:  # cross-hatch
        for c in range(-pw, pw + ph, gap):
            cv2.line(tile, (c, 0), (int(c - slope * ph), ph), color, thick, cv2.LINE_AA)


def screentone(tile: np.ndarray, rand) -> None:
    ph, pw = tile.shape[:2]
    gap = rand.randint(4, 12)
    radius = rand.randint(1, max(1, gap // 2 - 1))
    color = _gray(rand, 80, 180)
    for y in range(0, ph, gap):
        for x in range(0, pw, gap):
            cv2.circle(tile, (x, y), radius, color, -1, cv2.LINE_AA)


def linescreen(tile: np.ndarray, rand) -> None:
    ph, pw = tile.shape[:2]
    gap = rand.randint(3, 10)
    color = _gray(rand, 80, 180)
    if rand.random() < 0.5:
        for y in range(0, ph, gap):
            cv2.line(tile, (0, y), (pw, y), color, 1, cv2.LINE_AA)
    else:
        for x in range(0, pw, gap):
            cv2.line(tile, (x, 0), (x, ph), color, 1, cv2.LINE_AA)


_WHOLE = [strokes, shapes, borders]
_REGION = [hatching, screentone, linescreen]


def add_clutter(img: np.ndarray, rand) -> None:
    """Composite 1-3 random clutter elements onto `img` in place."""
    if rand.random() < 0.4:
        gradient(img, rand)
    for fn in rand.sample(_WHOLE + _REGION, k=rand.randint(1, 3)):
        if fn in _REGION:
            x, y, tile = _region(img, rand)
            fn(tile, rand)
            img[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
        else:
            fn(img, rand)
