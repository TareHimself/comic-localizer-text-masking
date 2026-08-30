"""Languages, font sets, and the round-robin that picks (language, font) pairs.

The set list comes from `<fonts_dir>/fonts.yaml` (see `fonts/fonts.yaml`).
`sep` is how a language's text splits into layout units: None = whitespace
(Latin), "" = per character (CJK). Per-character languages can be set
vertically.
"""

import itertools
import os
from dataclasses import dataclass

import yaml


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


def _language(entry: str | dict, default_sep: str | None) -> Language:
    if isinstance(entry, str):
        return Language(entry, default_sep)
    return Language(entry["code"], entry.get("sep", default_sep))


def load_language_sets(fonts_dir: str = "fonts") -> list[LanguageSet]:
    """Build the language/font sets from `<fonts_dir>/fonts.yaml`.

    Font files missing from `fonts_dir` are skipped; a set left with no fonts is
    dropped. Raises if the config is absent or nothing survives."""
    cfg_path = os.path.join(fonts_dir, "fonts.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found -- it lists the fonts to render with; "
            "supply the font files it names (see fonts/FONTS.md)"
        )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    sets: list[LanguageSet] = []
    for raw in cfg.get("sets", []):
        default_sep = raw.get("sep")
        langs = [_language(e, default_sep) for e in raw["languages"]]
        fonts = [
            p
            for n in raw.get("fonts", [])
            if os.path.isfile(p := os.path.join(fonts_dir, n))
        ]
        if langs and fonts:
            sets.append(LanguageSet(langs, fonts))

    if not sets:
        raise FileNotFoundError(
            f"none of the fonts named in {cfg_path} are present in "
            f"{fonts_dir!r} -- see fonts/FONTS.md"
        )
    return sets
