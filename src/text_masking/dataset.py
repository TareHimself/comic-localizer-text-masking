"""Build the mask dataset -- from flat colour surfaces or real cleaned pages.

    generate(out_dir, sets, count)                       # flat + clutter
    generate_from_backgrounds(cvlabel, out_dir, sets)    # real pages
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass

from . import cvlabel, parallel
from .clutter import add_clutter
from .geometry import clamp_box, random_boxes, round_to_8
from .languages import LanguageSet
from .page import AREA_LABELS, BUBBLE_LABELS, FLIPS, bubble_text_box, flip_page, free_text_boxes
from .parallel import Page
from .render import Box, TextSpec, compose
from .spec import SpecMaker, shard_seed
from .surface import flat_surface


def _spec_maker(sets, base_seed, shard, hyphenate) -> SpecMaker:
    return SpecMaker(sets, shard_seed(base_seed, shard), hyphenate)


def _fonts_langs(sets: list[LanguageSet]) -> tuple[list[str], list[str]]:
    langs, fonts = set(), set()
    for s in sets:
        langs.update(lang.code for lang in s.languages)
        fonts.update(os.path.basename(f) for f in s.fonts)
    return sorted(langs), sorted(fonts)


@dataclass
class FlatPages:
    """Producer: flat-colour surfaces with random boxes and procedural clutter."""

    sets: list[LanguageSet]
    base_seed: int
    hyphenate: bool
    surface_min: tuple[int, int]
    surface_max: tuple[int, int]
    max_boxes: int
    negative_prob: float
    clutter_prob: float

    def produce(self, shard: int, lo: int, hi: int) -> Iterator[Page]:
        maker = _spec_maker(self.sets, self.base_seed, shard, self.hyphenate)
        r = maker.rand
        for _ in range(lo, hi):
            w = round_to_8(r.randrange(self.surface_min[0], self.surface_max[0]))
            h = round_to_8(r.randrange(self.surface_min[1], self.surface_max[1]))
            surface, _ = flat_surface(w, h, r, maker.colors)

            over_art = r.random() < self.clutter_prob
            if over_art:
                add_clutter(surface, r)

            items: list[tuple[Box, TextSpec]] = []
            if r.random() >= self.negative_prob:
                for box in random_boxes(w, h, r.randint(1, self.max_boxes), r):
                    items.append((box, maker.for_box(surface, box, over_art=over_art)))

            yield compose(surface, items, r), [b for b, _ in items], r


@dataclass
class LabelledPages:
    """Producer: real cleaned pages, `Speech Bubble` / `Free Draw Area`
    annotations become the draw regions; each page is used at every flip."""

    cvlabel_path: str
    sets: list[LanguageSet]
    base_seed: int
    hyphenate: bool
    split: str | None
    repeats: int
    flips: tuple
    area_fill_prob: float
    bubble_labels: tuple = BUBBLE_LABELS
    area_labels: tuple = AREA_LABELS

    def produce(self, shard: int, lo: int, hi: int) -> Iterator[Page]:
        maker = _spec_maker(self.sets, self.base_seed, shard, self.hyphenate)
        r = maker.rand
        pages = cvlabel.read(self.cvlabel_path)
        if self.split is not None:
            pages = [p for p in pages if p.split == self.split]

        for _ in range(self.repeats):
            for base in pages[lo:hi]:
                for code in self.flips:
                    page = base if code is None else flip_page(base, code)
                    h, w = page.image.shape[:2]
                    items: list[tuple[Box, TextSpec]] = []

                    for a in page.by_label(*self.bubble_labels):
                        box = bubble_text_box(page.image, a, w, h)
                        if box:
                            items.append((box, maker.for_box(page.image, box, over_art=False)))

                    for a in page.by_label(*self.area_labels):
                        area = clamp_box(a.xyxy, w, h)
                        if not area:
                            continue
                        for box in free_text_boxes(area, r, self.area_fill_prob):
                            items.append((box, maker.for_box(page.image, box, over_art=True)))

                    yield compose(page.image, items, r), [b for b, _ in items], r


def generate(
    out_dir: str,
    sets: list[LanguageSet],
    count: int,
    *,
    seed: int = 0,
    surface_min: tuple[int, int] = (384, 384),
    surface_max: tuple[int, int] = (768, 1024),
    max_boxes: int = 4,
    negative_prob: float = 0.1,
    clutter_prob: float = 0.6,
    crops: bool = False,
    hyphenate: bool = True,
    workers: int = 0,
) -> None:
    producer = FlatPages(
        sets, seed, hyphenate, surface_min, surface_max,
        max_boxes, negative_prob, clutter_prob,
    )
    langs, fonts = _fonts_langs(sets)
    parallel.run(
        producer, out_dir,
        n_items=count, total=0 if crops else count,
        crops=crops, workers=workers, desc=out_dir,
        manifest={
            "kind": "synthetic-flat",
            "requested_pages": count,
            "crops": crops,
            "seed": seed,
            "clutter_prob": clutter_prob,
            "negative_prob": negative_prob,
            "max_boxes": max_boxes,
            "surface_min": list(surface_min),
            "surface_max": list(surface_max),
            "languages": langs,
            "fonts": fonts,
        },
    )


def generate_from_backgrounds(
    cvlabel_path: str,
    out_dir: str,
    sets: list[LanguageSet],
    *,
    seed: int = 0,
    repeats: int = 1,
    split: str | None = None,
    flips=FLIPS,
    area_fill_prob: float = 0.65,
    crops: bool = False,
    hyphenate: bool = True,
    workers: int = 0,
) -> None:
    flips = tuple(flips)
    pages = cvlabel.read(cvlabel_path)
    if split is not None:
        pages = [p for p in pages if p.split == split]
    if not pages:
        raise ValueError(f"no samples in {cvlabel_path} for split={split!r}")

    producer = LabelledPages(
        cvlabel_path, sets, seed, hyphenate, split, repeats, flips, area_fill_prob,
    )
    total = 0 if crops else len(pages) * repeats * len(flips)
    langs, fonts = _fonts_langs(sets)
    parallel.run(
        producer, out_dir,
        n_items=len(pages), total=total,
        crops=crops, workers=workers, desc=out_dir,
        manifest={
            "kind": "synthetic-on-pages",
            "backgrounds": cvlabel_path,
            "backgrounds_split": split,
            "pages": len(pages),
            "repeats": repeats,
            "flips": len(flips),
            "area_fill_prob": area_fill_prob,
            "crops": crops,
            "seed": seed,
            "languages": langs,
            "fonts": fonts,
        },
    )
