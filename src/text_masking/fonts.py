"""Font loading, hyphenation, and script-coverage character pools.

Faker gives natural character frequency, which barely exercises rare CJK
glyphs -- a mask model that never saw an unusual kanji may under-segment it.
`charset_text` samples more evenly across a script's range, filtered to what
the font can actually draw (its cmap), so we never render tofu boxes.
"""

import functools
import logging
import string

import pyphen
from fontTools.ttLib import TTFont
from PIL import ImageFont

# fontTools logs a warning per font whose `post` table has trailing bytes
# ("40 extra bytes in post.stringData array") -- harmless, and it floods the
# generation progress bar once per worker.
logging.getLogger("fontTools").setLevel(logging.ERROR)


@functools.lru_cache(maxsize=256)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@functools.lru_cache(maxsize=16)
def hyphenator(lang: str) -> pyphen.Pyphen:
    return pyphen.Pyphen(lang=lang)


# --------------------------------------------------------------------------- #
# character pools
# --------------------------------------------------------------------------- #


def _chars(lo: int, hi: int) -> list[str]:
    return [chr(c) for c in range(lo, hi + 1)]


LATIN = list(string.ascii_letters + string.digits + string.punctuation)
_KANA = _chars(0x3041, 0x3096) + _chars(0x30A1, 0x30FA)
_CJK_IDEO = _chars(0x4E00, 0x9FFF)  # CJK Unified Ideographs (BMP)
_HANGUL = _chars(0xAC00, 0xD7A3)  # syllable blocks
_CJK_PUNCT = list("、。，．・ー～！？：；「」『』（）")

POOLS: dict[str, list[str]] = {
    "latin": LATIN,
    "ja": _KANA + _CJK_IDEO + _CJK_PUNCT,
    "ko": _HANGUL + _CJK_PUNCT,
    "zh": _CJK_IDEO + _CJK_PUNCT,
}


def script_of(locale_code: str, sep: str | None) -> str:
    if sep != "":
        return "latin"
    if locale_code.startswith("ja"):
        return "ja"
    if locale_code.startswith("ko"):
        return "ko"
    return "zh"


@functools.lru_cache(maxsize=64)
def _supported(font_file: str) -> frozenset[str]:
    """Codepoints the font has a glyph for."""
    font = TTFont(font_file, fontNumber=0, lazy=True)
    return frozenset(chr(cp) for cp in font.getBestCmap())


@functools.lru_cache(maxsize=256)
def _drawable(font_file: str, script: str) -> tuple[str, ...]:
    supported = _supported(font_file)
    return tuple(c for c in POOLS[script] if c in supported)


def charset_text(
    font_file: str, locale_code: str, sep: str | None, n: int, rand
) -> str | None:
    """`n` glyphs sampled from `font_file`'s drawable subset of the locale's
    script. Latin is grouped into pseudo-words; CJK is a plain run. None if the
    font draws nothing in that script."""
    pool = _drawable(font_file, script_of(locale_code, sep))
    if not pool:
        return None
    chars = rand.choices(pool, k=max(1, n))
    if sep == "":
        return "".join(chars)
    words, i = [], 0
    while i < len(chars):
        w = rand.randint(2, 9)
        words.append("".join(chars[i : i + w]))
        i += w
    return " ".join(words)
