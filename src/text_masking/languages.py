"""Languages, font sets, and the round-robin that picks (language, font) pairs.

`sep` is how a language's text splits into layout units: None = whitespace
(Latin), "" = per character (CJK). Per-character languages can be set
vertically.
"""

import itertools
import os
from dataclasses import dataclass


@dataclass
class Language:
    code: str  # a Faker locale, e.g. "en_US", "ja"
    sep: str | None = None  # None = whitespace-separated, "" = per character


@dataclass
class LanguageSet:
    """One or more languages that share a font list."""

    languages: list[Language]
    fonts: list[str]


class LanguageCycler:
    """Round-robins languages; cycles each language's font list."""

    def __init__(self, sets: list[LanguageSet]):
        self.languages = [lang for s in sets for lang in s.languages]
        if not self.languages:
            raise ValueError("need at least one language")
        self._fonts = {
            lang.code: itertools.cycle(s.fonts)
            for s in sets
            for lang in s.languages
        }
        self._i = 0

    def next(self) -> tuple[Language, str]:
        lang = self.languages[self._i % len(self.languages)]
        self._i += 1
        return lang, next(self._fonts[lang.code])


_LATIN_FONTS = [
    "animeace2_bld.ttf",
    "animeace2_ital.ttf",
    "BlambotClassicBB-Bold.ttf",
    "BlambotClassicBB-Italic.ttf",
    "BlambotClassicBB.ttf",
    "Roboto-Light.ttf",
    "Roboto-Regular.ttf",
    "Roboto-Bold.ttf",
    "Roboto-Italic.ttf",
]
_JP_FONTS = [
    "07YasashisaAntique.otf",
    "07YasashisaBold.ttf",
    "kowaiFont.ttf",
    "LanobePOPv2.otf",
    "M-NijimiMincho.otf",
    "msmincho.ttf",
    "NikumaruFont.otf",
    "reiko.ttf",
    "YasashisaGothicBold-V2.otf",
    "SourceHanSans.ttc",
]


def default_languages(fonts_dir: str = "fonts") -> list[LanguageSet]:
    """The built-in en / ja / ko / zh set, fonts resolved against `fonts_dir`.

    Fonts not present on disk are skipped (see `fonts/FONTS.md`); a set with no
    surviving fonts is dropped. Raises if nothing is left."""

    def present(names: list[str]) -> list[str]:
        return [
            p for n in names
            if os.path.isfile(p := os.path.join(fonts_dir, n))
        ]

    candidates = [
        LanguageSet([Language("en_US"), Language("en_GB")], present(_LATIN_FONTS)),
        LanguageSet([Language("ja", "")], present(_JP_FONTS)),
        LanguageSet(
            [Language("ko", ""), Language("zh_CN", ""), Language("zh_TW", "")],
            present(["SourceHanSans.ttc"]),
        ),
    ]
    sets = [s for s in candidates if s.fonts]
    if not sets:
        raise FileNotFoundError(
            f"no fonts found under {fonts_dir!r} -- see fonts/FONTS.md"
        )
    return sets
