"""Colour helpers for text rendering."""

import colorsys

import numpy as np

RGB = tuple[int, int, int]

# minimum per-channel gap for text to count as readable against a background
MIN_CONTRAST = 60


class ColorGenerator:
    """Evenly spaced, pleasant RGB colours via golden-ratio hue stepping."""

    def __init__(self, seed: float = 0.0):
        self.hue = seed % 1.0

    def next_rgb(self) -> RGB:
        self.hue = (self.hue + 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(self.hue, 0.6, 0.85)
        return int(r * 255), int(g * 255), int(b * 255)


def luma(color: RGB) -> int:
    r, g, b = color
    return (r * 77 + g * 150 + b * 29) >> 8


def _contrasting_pole(color: RGB) -> RGB:
    """Black or white -- whichever reads against `color`."""
    return (0, 0, 0) if luma(color) >= 128 else (255, 255, 255)


def pick_text_color(bg_color: RGB, rand, colors: ColorGenerator) -> RGB:
    """A text colour guaranteed to read against a flat `bg_color`."""
    candidate = rand.choice([colors.next_rgb(), (0, 0, 0), (255, 255, 255)])
    if max(abs(a - b) for a, b in zip(candidate, bg_color)) >= MIN_CONTRAST:
        return candidate
    return _contrasting_pole(bg_color)


def pick_outline_color(
    text_color: RGB,
    background: np.ndarray,
    *,
    min_diff: int = 30,
    similar_fraction: float = 0.10,
    stride: int = 6,
) -> RGB | None:
    """Outline colour if `text_color` risks blending into `background`, else
    None. Samples every `stride`-th pixel to stay cheap."""
    sample = background[::stride, ::stride].astype(np.int16)
    diff = np.abs(sample - np.array(text_color, np.int16)).max(axis=2)
    if np.mean(diff < min_diff) <= similar_fraction:
        return None
    return _contrasting_pole(text_color)
