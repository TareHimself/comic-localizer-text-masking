"""Rendering: draw a `TextSpec` into a region, composite many onto a surface.

`compose(surface, [(box, TextSpec)])` is the core primitive -- it returns the
surface with all the text drawn plus a binary union mask. `surface` starts as a
flat colour and later becomes a real cleaned page.
"""

import math
import random
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .fonts import hyphenator, kana_text, load_font
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
    # draw a hand-drawn emphasis line (wavy / jagged) beside the text -- it is
    # part of the speech, so it also goes in the mask
    emphasis: bool = False
    # small ruby kana beside CJK text
    furigana: bool = False
    # rotate the whole block, degrees (SFX / hand-lettered wobble)
    rotate: float = 0.0
    # px added to every stroke: faux bold (shout / emphasis)
    stroke_bias: int = 0


def _squiggle(draw, draw_mask, box, color, mask_w: int, rand: random.Random, vertical: bool) -> None:
    x0, y0, x1, y1 = box
    axis = (y1 - y0) if vertical else (x1 - x0)
    if axis < 10:
        return
    jagged = rand.random() < 0.35
    amp = rand.uniform(2.0, 6.0)
    freq = rand.uniform(3.0, 9.0)
    phase = rand.uniform(0.0, 6.28)
    width = rand.randint(1, 3)
    steps = max(6, int(axis / 5))
    pts = []
    for i in range(steps + 1):
        t = i / steps
        off = (amp if i % 2 == 0 else -amp) if jagged else amp * math.sin(t * freq * math.pi + phase)
        if vertical:
            pts.append((0.5 * (x0 + x1) + off, y0 + t * axis))
        else:
            pts.append((x0 + t * axis, 0.5 * (y0 + y1) + off))
    draw.line(pts, fill=color, width=width, joint="curve")
    draw_mask.line(pts, fill=255, width=max(width, mask_w), joint="curve")


def place_text(
    area: np.ndarray, spec: TextSpec, rand: random.Random | None = None
) -> Sample | None:
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

    # text is drawn onto a transparent layer so it can be rotated / weight-
    # adjusted independently of the (real page) pixels underneath
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    draw_mask = ImageDraw.Draw(mask)
    fill = (*tuple(spec.text_color), 255)

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

    def put(pos: tuple[float, float], s: str, f=font, ms: int = mask_stroke) -> None:
        draw.text(
            pos, s, font=f, fill=fill,
            stroke_width=spec.outline_size,
            stroke_fill=(*tuple(outline), 255) if spec.outline_size else None,
        )
        draw_mask.text(
            pos, s, font=f, fill=255, stroke_width=ms, stroke_fill=255
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

    bx0, by0 = block_left, top
    bx1, by1 = block_left + wrap.width, top + wrap.height

    if spec.furigana and spec.sep == "" and rand is not None:
        kana = kana_text(spec.font_file, rand.randint(3, 9), rand)
        ruby = load_font(spec.font_file, max(6, int(size * 0.42)))
        rh = sum(ruby.getmetrics())
        if kana and spec.vertical and bx1 + rh + 2 < w:
            y = by0
            for ch in kana:
                put((bx1 + 1, y), ch, ruby, max(1, mask_stroke - 1))
                y += rh
        elif kana and not spec.vertical and by0 - rh - 1 > 0:
            x = bx0
            for ch in kana:
                put((x, by0 - rh - 1), ch, ruby, max(1, mask_stroke - 1))
                x += ruby.getlength(ch)

    if spec.emphasis and rand is not None:
        gap = rand.uniform(1.0, 8.0)
        if spec.vertical:
            cx = min(max((bx0 - gap) if rand.random() < 0.5 else (bx1 + gap), 3.0), w - 3.0)
            span = (cx - 4, by0, cx + 4, by1)
        else:
            cy = min(max((by0 - gap) if rand.random() < 0.5 else (by1 + gap), 3.0), h - 3.0)
            span = (bx0, cy - 4, bx1, cy + 4)
        _squiggle(draw, draw_mask, span, fill, mask_stroke, rand, spec.vertical)

    layer_np = np.array(layer)
    mask_np = np.array(mask)

    if spec.stroke_bias > 0:  # faux bold (shout / emphasis); never removes ink
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * spec.stroke_bias + 1,) * 2)
        layer_np, mask_np = cv2.dilate(layer_np, k), cv2.dilate(mask_np, k)

    if spec.rotate:
        cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
        layer_np = np.array(
            Image.fromarray(layer_np, "RGBA").rotate(
                spec.rotate, resample=Image.Resampling.BICUBIC, center=(cx, cy)
            )
        )
        mask_np = np.array(
            Image.fromarray(mask_np, "L").rotate(
                spec.rotate, resample=Image.Resampling.NEAREST, center=(cx, cy)
            )
        )

    if spec.mask_dilate > 0:
        k = 2 * spec.mask_dilate + 1
        mask_np = cv2.dilate(
            mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
    mask_np = np.where(mask_np > 127, np.uint8(255), np.uint8(0))
    if not mask_np.any():
        return None

    a = layer_np[..., 3:4].astype(np.float32) / 255.0
    out = area.astype(np.float32) * (1.0 - a) + layer_np[..., :3].astype(np.float32) * a
    return Sample(np.clip(out, 0, 255).astype(np.uint8), mask_np)


Box = tuple[int, int, int, int]


def compose(
    surface: np.ndarray,
    items: list[tuple[Box, TextSpec]],
    rand: random.Random | None = None,
) -> Sample:
    """Render every `(box, spec)` onto a copy of `surface` (RGB). `box` is
    `(x1, y1, x2, y2)`. Items whose text can't fit their box are skipped."""
    image = np.array(surface, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2), spec in items:
        patch = place_text(image[y1:y2, x1:x2], spec, rand)
        if patch is None:
            continue
        image[y1:y2, x1:x2] = patch.image
        region = mask[y1:y2, x1:x2]
        np.maximum(region, patch.mask, out=region)
    return Sample(image, mask)
