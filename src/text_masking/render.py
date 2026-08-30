"""Rendering: draw a `TextSpec` into a region, composite many onto a surface.

`compose(surface, [(box, TextSpec)])` is the core primitive -- it returns the
surface with all the text drawn plus a binary union mask. `surface` starts as a
flat colour and later becomes a real cleaned page.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .fonts import hyphenator, load_font
from .layout import fit_font


class Justification(Enum):
    Left = 0
    Center = 1
    Right = 2


class Alignment(Enum):
    Top = 0
    Center = 1
    Bottom = 2


@dataclass
class Sample:
    image: np.ndarray  # RGB uint8 (h, w, 3)
    mask: np.ndarray  # binary uint8 (h, w): 0 or 255


@dataclass
class TextSpec:
    """How to render one block of text (no geometry -- that's the box)."""

    text: str
    font_file: str
    sep: str | None = None
    text_color: tuple = (0, 0, 0)
    outline_color: tuple | None = None
    outline_size: int = 0
    justification: Justification = Justification.Center
    alignment: Alignment = Alignment.Center
    margin: int = 3
    line_spacing: float = 2.0
    target_size: int = 30
    # stack glyphs into right-to-left columns (manga-style vertical text)
    vertical: bool = False
    # pyphen language code (e.g. "en_US") to hyphenate overlong words, or None
    hyphen_lang: str | None = None
    # small, fixed margin so a model never crops the text (~2-3px past the ink)
    mask_pad: int = 2
    mask_dilate: int = 1


def place_text(area: np.ndarray, spec: TextSpec) -> Sample | None:
    """Render `spec.text` into `area` (an RGB view/array). Returns a Sample the
    same size as `area`, or None if the text can't be made to fit."""
    h, w = area.shape[:2]
    inner = (w - 2 * spec.margin, h - 2 * spec.margin)
    if min(inner) <= 0:
        return None

    fit = fit_font(
        spec.text,
        spec.font_file,
        inner,
        spec.target_size,
        spec.sep,
        spec.line_spacing + 2 * spec.outline_size,
        spec.vertical,
        hyphenator(spec.hyphen_lang)
        if spec.hyphen_lang and not spec.vertical
        else None,
    )
    if fit is None:
        return None
    size, wrap = fit
    font = load_font(spec.font_file, size)

    image = Image.fromarray(np.array(area, dtype=np.uint8))
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)

    if spec.alignment is Alignment.Top:
        top = float(spec.margin)
    elif spec.alignment is Alignment.Bottom:
        top = h - spec.margin - wrap.height
    else:
        top = (h - wrap.height) / 2

    if spec.justification is Justification.Left:
        block_left = float(spec.margin)
    elif spec.justification is Justification.Right:
        block_left = w - spec.margin - wrap.width
    else:
        block_left = (w - wrap.width) / 2

    outline = spec.outline_color or (255, 255, 255)
    mask_stroke = max(2, spec.outline_size + spec.mask_pad)

    def put(pos: tuple[float, float], s: str) -> None:
        draw.text(
            pos, s, font=font, fill=tuple(spec.text_color),
            stroke_width=spec.outline_size,
            stroke_fill=tuple(outline) if spec.outline_size else None,
        )
        draw_mask.text(
            pos, s, font=font, fill=255, stroke_width=mask_stroke, stroke_fill=255
        )

    if spec.vertical:
        glyph_h = sum(font.getmetrics())
        for col in wrap.lines:
            x = block_left + col.top
            y = top
            for glyph in col.words:
                put((x, y), glyph)
                y += glyph_h
    else:
        join = " " if spec.sep is None else spec.sep
        for line in wrap.lines:
            text = join.join(line.words)
            line_w = font.getlength(text)
            if spec.justification is Justification.Left:
                left = float(spec.margin)
            elif spec.justification is Justification.Right:
                left = w - spec.margin - line_w
            else:
                left = (w - line_w) / 2
            put((left, top + line.top), text)

    mask_np = np.array(mask)
    if spec.mask_dilate > 0:
        k = 2 * spec.mask_dilate + 1
        mask_np = cv2.dilate(
            mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
    mask_np = np.where(mask_np > 127, np.uint8(255), np.uint8(0))
    if not mask_np.any():
        return None
    return Sample(np.array(image), mask_np)


Box = tuple[int, int, int, int]


def compose(surface: np.ndarray, items: list[tuple[Box, TextSpec]]) -> Sample:
    """Render every `(box, spec)` onto a copy of `surface` (RGB). `box` is
    `(x1, y1, x2, y2)`. Items whose text can't fit their box are skipped."""
    image = np.array(surface, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2), spec in items:
        patch = place_text(image[y1:y2, x1:x2], spec)
        if patch is None:
            continue
        image[y1:y2, x1:x2] = patch.image
        region = mask[y1:y2, x1:x2]
        np.maximum(region, patch.mask, out=region)
    return Sample(image, mask)
