"""Console entry points (see [project.scripts] in pyproject.toml).

    tm-generate       synthetic mask dataset (flat, or on real cleaned pages)
    tm-prep-pages     inpaint labelled text out of pages (LaMa)
    tm-export-detect  labelled .cvlabel -> YOLO detection dataset
    tm-pack           pre-render a crop dataset into memmap packs
"""

from dataclasses import dataclass, field

from . import dataset
from . import export as _export
from . import pack as _pack
from .args import parse
from .clean import clean_pages as _clean_pages
from .languages import load_language_sets


# --------------------------------------------------------------------------- #
# tm-generate
# --------------------------------------------------------------------------- #


@dataclass
class GenerateConfig:
    out: str = "./data/mask_dataset"
    fonts_dir: str = "./fonts"
    # a cleaned .cvlabel to render onto; empty -> flat-colour synthetic surfaces
    backgrounds: str = ""
    train: int = 10_000  # flat mode: sample count
    val: int = 200
    repeats: int = 20  # background mode: synthetic-text passes per page
    crops: bool = False
    workers: int = 0


def generate() -> None:
    cfg = parse(GenerateConfig)
    sets = load_language_sets(cfg.fonts_dir)

    if cfg.backgrounds:
        dataset.generate_from_backgrounds(
            cfg.backgrounds, f"{cfg.out}/train", sets, seed=0,
            repeats=cfg.repeats, split="train", crops=cfg.crops, workers=cfg.workers,
        )
        dataset.generate_from_backgrounds(
            cfg.backgrounds, f"{cfg.out}/valid", sets, seed=30,
            repeats=1, split="valid", crops=cfg.crops, workers=cfg.workers,
        )
    else:
        for name, count, seed in (
            ("train", cfg.train, 0),
            ("valid", cfg.val, 30),
            ("test", cfg.val, 65),
        ):
            dataset.generate(
                f"{cfg.out}/{name}", sets, count, seed=seed,
                crops=cfg.crops, workers=cfg.workers,
            )


# --------------------------------------------------------------------------- #
# tm-prep-pages
# --------------------------------------------------------------------------- #


@dataclass
class PrepPagesConfig:
    labels: str  # input labelled .cvlabel
    out: str  # output cleaned .cvlabel
    cleaner_model: str  # LaMa torchscript
    device: str = "cuda:0"
    pad: int = 6


def prep_pages() -> None:
    cfg = parse(PrepPagesConfig)
    _clean_pages(cfg.labels, cfg.out, cfg.cleaner_model, device=cfg.device, pad=cfg.pad)


# --------------------------------------------------------------------------- #
# tm-export-detect
# --------------------------------------------------------------------------- #


@dataclass
class ExportDetectConfig:
    labels: str  # original labelled .cvlabel (not the cleaned one)
    out: str = "./data/detect_dataset"


def export_detect() -> None:
    cfg = parse(ExportDetectConfig)
    _export.export_detect(cfg.labels, cfg.out)


# --------------------------------------------------------------------------- #
# tm-pack
# --------------------------------------------------------------------------- #


@dataclass
class PackConfig:
    data: list[str] = field(default_factory=list)  # crop dataset roots
    out: str = "./data/packed"
    imgsz: int = 384
    workers: int = 0


def pack() -> None:
    cfg = parse(PackConfig)
    if not cfg.data:
        raise SystemExit("--data <root> [root ...]")
    _pack.pack(cfg.data, cfg.out, cfg.imgsz, cfg.workers)
