"""Flat-colour surfaces for synthetic (non-page) samples."""

import numpy as np

from .color import RGB, ColorGenerator


def flat_surface(
    width: int, height: int, rand, colors: ColorGenerator
) -> tuple[np.ndarray, RGB]:
    """An RGB surface of one colour + that colour: white / black / random."""
    roll = rand.random()
    if roll < 0.15:
        color = (255, 255, 255)
    elif roll < 0.30:
        color = (0, 0, 0)
    elif rand.random() < 0.5:
        color = colors.next_rgb()
    else:
        color = tuple(rand.randint(0, 255) for _ in range(3))
    return np.full((height, width, 3), color, dtype=np.uint8), color
