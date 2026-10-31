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

# punctuation is a tiny fraction of a script's glyphs but common in real text
# and hard to segment (small, isolated), so `charset_text` samples it
# separately rather than letting it drown in the ideograph pool
_CJK_PUNCT = list("、。，．・ー～！？：；「」『』（）【】〔〕…‥―—〜々〆♪＂")
_LATIN_PUNCT = list(string.punctuation) + ["...", "--", "!?", "?!", "!!", "—", "…"]

POOLS: dict[str, list[str]] = {
    "latin": list(string.ascii_letters + string.digits),
    "ja": _KANA + _CJK_IDEO,
    "ko": _HANGUL,
    "zh": _CJK_IDEO,
}
PUNCT: dict[str, list[str]] = {
    "latin": _LATIN_PUNCT,
    "ja": _CJK_PUNCT,
    "ko": _CJK_PUNCT,
    "zh": _CJK_PUNCT,
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


@functools.lru_cache(maxsize=512)
def _drawable(font_file: str, script: str, punct: bool = False) -> tuple[str, ...]:
    supported = _supported(font_file)
    tokens = PUNCT[script] if punct else POOLS[script]
    return tuple(t for t in tokens if all(c in supported for c in t))


def kana_text(font_file: str, n: int, rand) -> str:
    """A short hiragana/katakana run the font can draw -- used for furigana."""
    supported = _supported(font_file)
    pool = [c for c in _KANA if c in supported]
    return "".join(rand.choices(pool, k=max(1, n))) if pool else ""


def charset_text(
    font_file: str,
    locale_code: str,
    sep: str | None,
    n: int,
    rand,
    punct_prob: float = 0.15,
) -> str | None:
    """`n` glyphs sampled from `font_file`'s drawable subset of the locale's
    script, with `punct_prob` of each pick swapped for a punctuation mark (and
    ~1 run in 12 mostly punctuation). Latin is grouped into pseudo-words; CJK is
    a plain run. None if the font draws nothing in that script."""
    script = script_of(locale_code, sep)
    pool = _drawable(font_file, script)
    marks = _drawable(font_file, script, punct=True)
    if not pool:
        return None

    p = 0.7 if (marks and rand.random() < 1 / 12) else punct_prob
    chars = [
        rand.choice(marks) if marks and rand.random() < p else rand.choice(pool)
        for _ in range(max(1, n))
    ]
    if sep == "":
        return "".join(chars)
    words, i = [], 0
    while i < len(chars):
        w = rand.randint(2, 9)
        words.append("".join(chars[i : i + w]))
        i += w
    return " ".join(words)
