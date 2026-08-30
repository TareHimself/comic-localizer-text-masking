"""Word-wrapping and font-size fitting.

`fit_font` binary-searches the largest font size at which text wraps inside a
box; `wrap_lines` / `wrap_vertical` do the wrapping for horizontal and
manga-style vertical (right-to-left columns) text.
"""

from dataclasses import dataclass

import pyphen
from PIL import ImageFont

from .fonts import load_font


@dataclass
class WrapLine:
    words: list[str]
    top: float  # offset within the block along the wrap axis
    #            (y for a horizontal line, x for a vertical column)


@dataclass
class WrapResult:
    lines: list[WrapLine]
    width: float
    height: float


def _break_word(
    word: str,
    font: ImageFont.FreeTypeFont,
    first_width: float,
    line_width: float,
    dic: pyphen.Pyphen,
) -> list[str] | None:
    """Split `word` at hyphenation points into line pieces. Piece 0 must fit
    `first_width`; the rest fit `line_width`. Non-final pieces get a trailing
    "-". None if it can't be broken small enough."""
    stops = [*dic.positions(word), len(word)]
    pieces: list[str] = []
    start, budget = 0, first_width
    while start < len(word):
        chosen = None
        for stop in stops:
            if stop <= start:
                continue
            piece = word[start:stop] + ("" if stop == len(word) else "-")
            if font.getlength(piece) <= budget:
                chosen = (stop, piece)
            else:
                break
        if chosen is None:
            return None
        start, piece = chosen
        pieces.append(piece)
        budget = line_width
    return pieces


def wrap_lines(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: float,
    line_spacing: float,
    sep: str | None,
    dic: pyphen.Pyphen | None = None,
) -> WrapResult | None:
    """Greedy word-wrap. `sep` splits `text` into units: None -> whitespace,
    "" -> characters, otherwise that literal separator. With `dic`, words too
    wide for a line are hyphenated (whitespace text only); without it, such a
    word makes this return None so the caller can shrink the font."""
    units = [u for u in (list(text) if sep == "" else text.split(sep)) if u != ""]
    if not units:
        return None

    gap = font.getlength(" ") if sep is None else 0.0
    ascent, descent = font.getmetrics()
    line_height = ascent + descent

    rows: list[list[str]] = [[]]
    x = 0.0
    for unit in units:
        w = font.getlength(unit)
        remaining = (max_width - x) if rows[-1] else max_width

        if w <= remaining:
            rows[-1].append(unit)
            x += w + gap
            continue

        if dic is not None and sep is None:
            pieces = _break_word(unit, font, remaining, max_width, dic)
            if pieces is None and rows[-1]:
                pieces = _break_word(unit, font, max_width, max_width, dic)
                if pieces is not None:
                    rows.append([])
            if pieces is not None:
                for i, piece in enumerate(pieces):
                    if i > 0:
                        rows.append([])
                    rows[-1].append(piece)
                    x = font.getlength(piece) + gap
                continue

        if w <= max_width:
            rows.append([unit])
            x = w + gap
            continue

        return None

    def row_width(row: list[str]) -> float:
        return sum(font.getlength(u) for u in row) + gap * max(len(row) - 1, 0)

    step = line_height + line_spacing
    lines = [WrapLine(row, i * step) for i, row in enumerate(rows)]
    height = len(rows) * line_height + (len(rows) - 1) * line_spacing
    return WrapResult(lines, max(map(row_width, rows)), height)


def wrap_vertical(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_height: float,
    col_spacing: float,
) -> WrapResult | None:
    """Stack glyphs top-to-bottom into columns; new columns go leftward (RTL) --
    the manga convention. Units are individual characters."""
    units = [c for c in text if c != "\n"]
    if not units:
        return None

    ascent, descent = font.getmetrics()
    glyph_h = ascent + descent
    if glyph_h > max_height:
        return None
    col_w = max(font.getlength(u) for u in units)

    cols: list[list[str]] = [[]]
    y = 0.0
    for u in units:
        if cols[-1] and y + glyph_h > max_height:
            cols.append([])
            y = 0.0
        cols[-1].append(u)
        y += glyph_h

    step = col_w + col_spacing
    width = len(cols) * col_w + (len(cols) - 1) * col_spacing
    height = max(len(c) for c in cols) * glyph_h
    # column 0 is the rightmost; its left edge sits at width - col_w
    lines = [WrapLine(col, width - col_w - i * step) for i, col in enumerate(cols)]
    return WrapResult(lines, width, height)


def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    box: tuple[int, int],
    line_spacing: float,
    sep: str | None,
    vertical: bool,
    dic: pyphen.Pyphen | None = None,
) -> WrapResult | None:
    box_w, box_h = box
    if vertical:
        return wrap_vertical(text, font, box_h, line_spacing)
    return wrap_lines(text, font, box_w, line_spacing, sep, dic)


def fit_font(
    text: str,
    font_path: str,
    box: tuple[int, int],
    target_size: int,
    sep: str | None,
    line_spacing: float,
    vertical: bool = False,
    dic: pyphen.Pyphen | None = None,
    min_size: int = 8,
) -> tuple[int, WrapResult] | None:
    """Largest size in [min_size, target_size] at which `text` fits inside
    `box` (w, h). None if it doesn't fit even at `min_size`."""
    box_w, box_h = box
    best = None
    lo, hi = min_size, max(min_size, target_size)
    while lo <= hi:
        size = (lo + hi) // 2
        wrap = wrap_text(
            text, load_font(font_path, size), box, line_spacing, sep, vertical, dic
        )
        if wrap is not None and wrap.width <= box_w and wrap.height <= box_h:
            best = (size, wrap)
            lo = size + 1
        else:
            hi = size - 1
    return best
