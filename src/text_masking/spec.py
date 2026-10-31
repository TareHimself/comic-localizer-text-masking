"""Turn a box (and the pixels behind it) into a randomised `TextSpec`.

`SpecMaker` holds the per-run RNG / colour / language state so a box only needs
`surface` and `over_art`; everything else -- text, font, colour, outline,
orientation -- is drawn from it.
"""

import random

import numpy as np
from faker import Faker

from .color import ColorGenerator, luma, pick_outline_color, pick_text_color
from .fonts import charset_text
from .languages import Language, LanguageCycler, LanguageSet
from .render import Alignment, Justification, TextSpec

_DERIVED_SEED = 100_003


def shard_seed(base_seed: int, shard: int) -> int:
    return base_seed * _DERIVED_SEED + shard


def _text_length(area: int, rand: random.Random, big: bool = False) -> tuple[bool, bool, int]:
    if big:  # a few huge glyphs filling the box -- SFX, shouts, titles
        return False, False, rand.randint(1, 5)
    small, medium = area < 40_000, area < 120_000
    n = (
        rand.randint(2, 8)
        if small
        else rand.randint(8, 30)
        if medium
        else rand.randint(24, 90)
    )
    return small, medium, n


def random_text(
    faker: Faker,
    lang: Language,
    font_file: str,
    box_w: int,
    box_h: int,
    rand: random.Random,
    charset_prob: float = 0.4,
    big: bool = False,
) -> str:
    """Text whose length roughly suits the box. `charset_prob` of the time a
    random-glyph run (even coverage of the script + font), otherwise natural
    Faker text (realistic layout / clustering). `big` forces a short string so
    the fitter can blow it up to fill the box."""
    small, medium, n = _text_length(box_w * box_h, rand, big)

    if big:
        text = charset_text(font_file, lang.code, lang.sep, n, rand)
        if text:
            return text
        return " ".join(faker.word() for _ in range(rand.randint(1, 3)))

    if rand.random() < charset_prob:
        text = charset_text(font_file, lang.code, lang.sep, n, rand)
        if text:
            return text

    if small:
        return faker.sentence(nb_words=rand.randint(1, 3))
    if medium and rand.random() < 0.6:
        return faker.sentence(nb_words=rand.randint(3, 8))
    return rand.choice([faker.sentence, faker.address, faker.company])()


def _vertical(lang: Language, box, rand: random.Random) -> bool:
    """Per-character scripts can be set vertically (manga convention). Japanese
    genuinely uses both; ko / zh are mostly horizontal. Tall boxes lean
    vertical."""
    if lang.sep != "":
        return False
    x1, y1, x2, y2 = box
    base = 0.55 if lang.code.startswith("ja") else 0.15
    tall = (y2 - y1) > (x2 - x1) * 1.3
    return rand.random() < (base + 0.2 if tall else base - 0.15)


class SpecMaker:
    def __init__(self, sets: list[LanguageSet], seed: int, hyphenate: bool = True):
        Faker.seed(seed)
        self.rand = random.Random(seed)
        self.colors = ColorGenerator(seed)
        self.cycler = LanguageCycler(sets)
        self.fakers = {lang.code: Faker(lang.code) for lang in self.cycler.languages}
        self.hyphenate = hyphenate

    def for_box(self, surface: np.ndarray, box, *, over_art: bool) -> TextSpec:
        x1, y1, x2, y2 = box
        patch = surface[max(0, y1):y2, max(0, x1):x2]
        r = self.rand

        if patch.size:
            bg = tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
        else:
            bg = (255, 255, 255)

        lang, font = self.cycler.next()
        big = r.random() < 0.22  # oversized text -- rare, but the model must see it
        text_color = pick_text_color(bg, r, self.colors)
        outline = pick_outline_color(text_color, patch) if patch.size else None
        # free text sits on artwork -> nearly always outlined, like real SFX
        if over_art and outline is None and r.random() < 0.9:
            outline = (0, 0, 0) if luma(text_color) >= 128 else (255, 255, 255)

        # at big sizes a 1-3px outline is invisible but still fattens the mask,
        # so go either properly thick (SFX-style) or none
        if big:
            outline_size = r.randint(4, 12) if (outline and r.random() < 0.5) else 0
        else:
            outline_size = r.randint(1, 3) if outline else 0

        return TextSpec(
            text=random_text(
                self.fakers[lang.code], lang, font, x2 - x1, y2 - y1, r, big=big
            ),
            font_file=font,
            sep=lang.sep,
            text_color=text_color,
            outline_color=outline if outline_size else None,
            outline_size=outline_size,
            justification=r.choice(list(Justification)),
            alignment=r.choice(list(Alignment)),
            margin=r.randint(0, 4),
            line_spacing=r.randint(1, 3),
            target_size=r.randint(64, 140) if big else r.randint(18, 56),
            vertical=_vertical(lang, box, r),
            hyphen_lang=lang.code if self.hyphenate and lang.sep is None else None,
            # big glyphs: keep the mask tight so complex kanji counters stay open
            mask_pad=1 if big else 2,
            mask_dilate=0 if big else 1,
            # a hand-drawn wavy / jagged emphasis line beside the text
            emphasis=r.random() < 0.12,
            # ruby kana beside CJK
            furigana=lang.sep == "" and r.random() < 0.25,
            # SFX gets a real tilt; bubble text a slight hand-lettered wobble
            rotate=(
                r.uniform(-27, 27) if big and r.random() < 0.6
                else r.uniform(-4, 4) if r.random() < 0.3
                else 0.0
            ),
            # faux bold for shouting / emphasis (dilate only -- never erases ink)
            stroke_bias=r.choice([0, 0, 0, 0, 0, 1, 1, 2]),
        )
