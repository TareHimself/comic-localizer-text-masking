"""Export a trained run to the HuggingFace Hub.

One model repo gets three artefacts plus a card:

    config.json + model.safetensors   pristine smp weights (`smp.from_pretrained`)
    model.pt                          TorchScript: letterboxed uint8 RGB -> text
                                      probability, with /255 + ImageNet
                                      normalisation + sigmoid baked into the graph
    tm_meta.json                      imgsz, the letterbox contract, threshold,
                                      val metrics, languages

Geometry (aspect-preserving letterbox to `imgsz`, pad 0, and its inverse) stays
in the caller -- it is a few lines of array work and must match training's
`geometry.letterbox`.
"""

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from .seg.data import IMAGENET_MEAN, IMAGENET_STD
from .seg.model import build_model


class _Inference(nn.Module):
    """Wrap the trained net so the traced graph takes raw letterboxed uint8 RGB
    `(B, 3, S, S)` and returns a text probability map `(B, 1, S, S)`."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core.eval()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.to(self.mean.dtype) / 255.0 - self.mean) / self.std
        return self.core(x).sigmoid()


def _resolve_run(spec: str, runs_dir: str) -> Path:
    if os.path.isdir(spec) and (Path(spec) / "checkpoints" / "best.pt").is_file():
        return Path(spec)
    if os.path.isfile(spec):  # a best.pt itself
        return Path(spec).parent.parent
    if "latest" in spec and os.path.isdir(runs_dir):
        for d in sorted(os.listdir(runs_dir), reverse=True):
            if (Path(runs_dir) / d / "checkpoints" / "best.pt").is_file():
                return Path(runs_dir) / d
    raise SystemExit(f"can't resolve {spec!r} to a run with a checkpoints/best.pt")


def _languages(run_path: Path) -> list[str]:
    cfg_path = run_path / "config.json"
    if not cfg_path.is_file():
        return []
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    langs: set[str] = set()
    for sources in cfg.get("data_sources", {}).values():
        for src in sources:
            langs.update(src.get("manifest", {}).get("languages", []))
    return sorted(langs)


def _card(repo: str, github: str, imgsz: int, metrics: dict, languages: list[str]) -> str:
    iso = sorted({c.split("_")[0] for c in languages})
    fm = "\n".join(f"  - {c}" for c in iso)
    m = metrics
    return f"""---
license: mit
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
tags:
  - comics
  - manga
  - text-segmentation
language:
{fm}
---

# {repo.split("/")[-1]}

Binary **text-mask segmenter** for the
[comic-localizer](https://github.com/TareHimself/comic-localizer) cleaning
pipeline. It runs on a detector's text-region crop and returns a per-pixel
"text vs not-text" mask that feeds LaMa inpainting. Locating and grouping text
is the detector's job, not this model's.

Training data (see the [training repo]({github})): synthetic text rendered onto
(a) procedurally generated flat surfaces with synthetic clutter and (b) real
cleaned comic pages, framed as detector-style crops. Text is Latin, Japanese
(kana + kanji, horizontal and vertical), Korean, and Chinese, with a large
fraction of random-glyph runs so rare characters are covered. No source imagery
is redistributed.

Validation (held-out synthetic + real crops): IoU {m.get('iou', 0):.3f},
precision {m.get('precision', 0):.3f}, recall {m.get('recall', 0):.3f}.

## Use

```python
import json, torch
from huggingface_hub import hf_hub_download

meta = json.load(open(hf_hub_download("{repo}", "tm_meta.json")))
model = torch.jit.load(hf_hub_download("{repo}", "model.pt")).eval()
S = meta["imgsz"]

# letterbox `rgb` (H,W,3 uint8) into an SxS square, pad 0, keep the paste box
# ... then:
x = torch.from_numpy(square).permute(2, 0, 1).unsqueeze(0)      # (1,3,S,S) uint8
prob = model(x)[0, 0].numpy()                                   # (S,S) float
mask = (prob > meta["threshold"]).astype("uint8") * 255
# crop the paste box back out and resize to the original size
```

Or load the raw weights with `segmentation-models-pytorch`:

```python
import segmentation_models_pytorch as smp
model = smp.from_pretrained("{repo}")   # normalisation NOT baked in
```

`model.pt` has `/255`, ImageNet normalisation, and the final sigmoid baked into
the graph; it expects letterboxed **uint8** RGB. `model.safetensors` is the
pristine network.
"""


def push(
    run: str,
    repo: str,
    *,
    runs_dir: str = "./runs",
    threshold: float = 0.5,
    tag: str = "",
    private: bool = False,
    token: str | None = None,
) -> None:
    from huggingface_hub import HfApi

    run_path = _resolve_run(run, runs_dir)
    ck = torch.load(
        run_path / "checkpoints" / "best.pt", map_location="cpu", weights_only=False
    )
    imgsz, metrics = ck["imgsz"], ck.get("metrics", {})
    languages = _languages(run_path)

    core = build_model(ck["arch"], ck["encoder"], pretrained=False)
    core.load_state_dict(ck["model"])
    core.eval()

    out = Path(tempfile.mkdtemp())
    core.save_pretrained(out)  # config.json + model.safetensors

    wrapper = _Inference(core).eval()
    example = torch.randint(0, 256, (1, 3, imgsz, imgsz), dtype=torch.uint8)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
        check = torch.randint(0, 256, (1, 3, imgsz, imgsz), dtype=torch.uint8)
        if not torch.allclose(traced(check), wrapper(check), atol=1e-4):
            raise RuntimeError("traced model disagrees with the eager model")
    torch.jit.save(traced, str(out / "model.pt"))

    (out / "tm_meta.json").write_text(
        json.dumps(
            {
                "imgsz": imgsz,
                "input": "letterboxed uint8 RGB (1,3,imgsz,imgsz), pad 0",
                "output": "float32 text probability (1,1,imgsz,imgsz)",
                "normalization": "baked into model.pt",
                "threshold": threshold,
                "arch": ck["arch"],
                "encoder": ck["encoder"],
                "metrics": {k: round(float(v), 5) for k, v in metrics.items()},
                "languages": languages,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        _card(repo, "https://github.com/TareHimself/comic-localizer-text-masking",
              imgsz, metrics, languages),
        encoding="utf-8",
    )

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo, folder_path=str(out), commit_message=f"from {run_path.name}"
    )
    if tag:
        try:  # move the tag onto this push if it already exists
            api.delete_tag(repo, tag=tag, repo_type="model")
        except Exception:
            pass
        api.create_tag(repo, tag=tag, repo_type="model")
    print(
        f"pushed {run_path.name} -> https://huggingface.co/{repo}"
        + (f"  (tag {tag})" if tag else "")
    )
