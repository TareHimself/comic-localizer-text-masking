"""Train the text-mask segmenter. Each run gets its own `runs/<run>/` dir.

    tm-train --data ./data/packed_384 --imgsz 384 --encoder resnet18 --name syn-real
    tm-train --smoke
    tm-train --wandb
"""

import json
import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..args import parse
from .data import MaskDataset, denormalize, loader
from .model import build_model, dice_bce_loss
from .run import RunDir

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data", "mask_dataset"))


@dataclass
class Config:
    # one or more dataset roots (raw crop dirs or tm-pack packs); splits concat
    data: list[str] = field(default_factory=lambda: [DEFAULT_DATA])
    name: str = ""  # run name; runs go to runs/<timestamp>_<name>/
    runs_dir: str = "./runs"
    arch: str = "unet"  # unet | fpn
    encoder: str = "resnet18"
    # warm-start weights from a checkpoint (a run's best.pt, or "latest" for the
    # newest run) -- fresh optimizer / schedule / epoch count, just the weights
    init: str = ""
    imgsz: int = 384
    batch: int = 32
    epochs: int = 80
    patience: int = 5  # stop if val IoU stalls this many epochs (0 = off)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 12
    device: str = "cuda:0"
    smoke: bool = False  # fast preset: small subset, fewer epochs
    limit: int = 0  # cap train samples (0 = all)
    wandb: bool = False
    wandb_project: str = "text-mask-seg"


def _resolve_init(spec: str, runs_dir: str) -> str:
    """A checkpoint path, a run dir, or "latest" -> a concrete .pt path."""
    if os.path.isfile(spec):
        return spec
    if os.path.isdir(spec):
        return os.path.join(spec, "checkpoints", "best.pt")
    if "latest" in spec and os.path.isdir(runs_dir):
        cands = sorted(
            os.path.join(runs_dir, d, "checkpoints", "best.pt")
            for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d))
        )
        for p in reversed(cands):  # newest run that actually has a best.pt
            if os.path.isfile(p):
                return p
    raise SystemExit(f"--init: can't resolve {spec!r} to a checkpoint")


def _load_provenance(roots: list[str]) -> dict:
    """{split: [source entries]} read back from each pack's meta.json."""
    prov: dict = {}
    for split in ("train", "valid"):
        for r in roots:
            meta = os.path.join(r, split, "meta.json")
            if os.path.exists(meta):
                with open(meta, encoding="utf-8") as f:
                    srcs = json.load(f).get("sources", [])
                for s in srcs:
                    # pre-provenance packs stored sources as bare path strings
                    s = s if isinstance(s, dict) else {"root": s}
                    man = s.get("manifest")
                    if man and "fonts" in man:  # keep the count, not 19 filenames
                        man["n_fonts"] = len(man.pop("fonts"))
                    prov.setdefault(split, []).append(s)
    return prov


def _summarize_provenance(prov: dict) -> str:
    """One human line per split: `train: 15228 flat + 13411 cleaned.cvlabel[train]x30`."""
    lines = []
    for split, sources in prov.items():
        parts = []
        for src in sources:
            man, n = src.get("manifest", {}), src.get("n", "?")
            if man.get("kind") == "synthetic-on-pages":
                bg = os.path.basename(man.get("backgrounds", "?"))
                parts.append(
                    f"{n} {bg}[{man.get('backgrounds_split')}]x{man.get('repeats')}"
                )
            elif man.get("kind") == "synthetic-flat":
                parts.append(f"{n} flat")
            else:
                parts.append(f"{n} {src.get('root', '?')}")
        lines.append(f"{split}: " + " + ".join(parts))
    return "\n".join(lines)


def train_epoch(model, dl, opt, scaler, loss_fn, device, amp, desc, on_step=None) -> float:
    model.train()
    total = 0.0
    for img, mask in tqdm(dl, desc=desc):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp):
            loss = loss_fn(model(img), mask)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        if on_step is not None:
            on_step(loss.item())
    return total / len(dl)


@torch.no_grad()
def evaluate(model, dl, device, amp) -> dict:
    """Pixel IoU / precision / recall over the whole loader."""
    model.eval()
    tp = fp = fn = 0.0
    for img, mask in dl:
        img, mask = img.to(device), mask.to(device)
        with torch.amp.autocast(device.type, enabled=amp):
            pred = (model(img).sigmoid() > 0.5).float()
        tp += (pred * mask).sum().item()
        fp += (pred * (1 - mask)).sum().item()
        fn += ((1 - pred) * mask).sum().item()
    return {
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
        "precision": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall": tp / (tp + fn) if (tp + fn) else 1.0,
    }


@torch.no_grad()
def overlay_panels(model, dl, device, amp, count=6) -> list[np.ndarray]:
    """`image | prediction | ground-truth` strips (BGR) for the first val batch."""
    model.eval()
    img, mask = next(iter(dl))
    with torch.amp.autocast(device.type, enabled=amp):
        pred = (model(img.to(device)).sigmoid() > 0.5).cpu().numpy()[:, 0]
    gt = mask.numpy()[:, 0]
    out = []
    for i in range(min(count, len(img))):
        rgb = denormalize(img[i])
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        p = cv2.cvtColor((pred[i] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        g = cv2.cvtColor((gt[i] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        sep = np.full((base.shape[0], 3, 3), 255, np.uint8)
        out.append(np.hstack([base, sep, p, sep, g]))
    return out


def main() -> None:
    cfg = parse(Config)
    if cfg.smoke:
        cfg.epochs, cfg.limit = 12, 800

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available -> CPU (slow).")
        cfg.device = "cpu"
    device = torch.device(cfg.device)
    amp = device.type == "cuda"

    # resolve before the new run dir exists, so "latest" can't pick this run
    init_ckpt = _resolve_init(cfg.init, cfg.runs_dir) if cfg.init else ""

    run = RunDir(cfg.runs_dir, cfg.name)
    run.log(f"run -> {run.path}")

    cfg_dict = asdict(cfg)
    prov = _load_provenance(cfg.data)
    if prov:
        cfg_dict["data_sources"] = prov
        cfg_dict["data_summary"] = _summarize_provenance(prov)
        for line in cfg_dict["data_summary"].splitlines():
            run.log(line)
    run.save_config(cfg_dict)

    wb = None
    if cfg.wandb:
        import wandb

        # dir=run.path -> wandb's local tree lands in runs/<run>/wandb/, so
        # deleting the run folder removes its wandb data too
        wb = wandb.init(
            project=cfg.wandb_project,
            name=cfg.name or None,
            config=cfg_dict,
            dir=str(run.path),
        )
        (run.path / "wandb_url.txt").write_text(wb.url or "", encoding="utf-8")

    train_dl = loader(
        MaskDataset(cfg.data, "train", cfg.imgsz, True, cfg.limit),
        cfg.batch, cfg.workers, train=True,
    )
    valid_dl = loader(
        MaskDataset(cfg.data, "valid", cfg.imgsz, False),
        cfg.batch, cfg.workers, train=False,
    )
    run.log(f"train {len(train_dl.dataset)}  valid {len(valid_dl.dataset)}")

    model = build_model(cfg.arch, cfg.encoder, pretrained=not init_ckpt).to(device)
    if init_ckpt:
        ck = torch.load(init_ckpt, map_location=device, weights_only=False)
        if (ck.get("arch"), ck.get("encoder")) != (cfg.arch, cfg.encoder):
            raise SystemExit(
                f"--init is {ck.get('arch')}/{ck.get('encoder')}, "
                f"this run is {cfg.arch}/{cfg.encoder}"
            )
        model.load_state_dict(ck["model"])
        run.log(
            f"init from {init_ckpt}  (was epoch {ck.get('epoch')}, "
            f"val IoU {ck.get('metrics', {}).get('iou', float('nan')):.4f})"
        )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn = dice_bce_loss()

    def save(name: str, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "arch": cfg.arch,
                "encoder": cfg.encoder,
                "imgsz": cfg.imgsz,
                "epoch": epoch,
                "metrics": metrics,
                "config": asdict(cfg),
            },
            run.checkpoint(name),
        )

    # step-0 baseline: eval the untrained model so wandb has a curve (not a
    # lone point) and metrics.csv / samples exist before epoch 1 finishes
    step = 0

    def on_step(loss_v: float) -> None:
        nonlocal step
        step += 1
        if wb is not None and (step == 1 or step % 20 == 0):
            wb.log({"train/loss": loss_v, "lr": sched.get_last_lr()[0]}, step=step)

    m0 = evaluate(model, valid_dl, device, amp)
    run.log(
        f"epoch 0 (init): val IoU {m0['iou']:.4f}  "
        f"P {m0['precision']:.4f}  R {m0['recall']:.4f}"
    )
    run.log_metrics(
        {"epoch": 0, "train_loss": "", "lr": cfg.lr, **{f"val_{k}": v for k, v in m0.items()}}
    )
    run.save_overlays(0, overlay_panels(model, valid_dl, device, amp))
    if wb is not None:
        wb.log({"lr": cfg.lr, **{f"val/{k}": v for k, v in m0.items()}}, step=0)

    best_iou, stalled, reason = 0.0, 0, "completed"
    try:
        for epoch in range(1, cfg.epochs + 1):
            loss = train_epoch(
                model, train_dl, opt, scaler, loss_fn, device, amp,
                f"epoch {epoch}/{cfg.epochs}", on_step,
            )
            sched.step()
            m = evaluate(model, valid_dl, device, amp)
            improved = m["iou"] > best_iou
            stalled = 0 if improved else stalled + 1

            run.log(
                f"epoch {epoch}: loss {loss:.4f}  val IoU {m['iou']:.4f}  "
                f"P {m['precision']:.4f}  R {m['recall']:.4f}"
                + ("  *" if improved else f"  ({stalled}/{cfg.patience})")
            )
            run.log_metrics(
                {"epoch": epoch, "train_loss": loss, "lr": sched.get_last_lr()[0], **{f"val_{k}": v for k, v in m.items()}}
            )
            save("last", epoch, m)

            if improved:
                best_iou = m["iou"]
                save("best", epoch, m)
                run.save_overlays(epoch, overlay_panels(model, valid_dl, device, amp))

            if wb is not None:
                wb.log({f"val/{k}": v for k, v in m.items()}, step=step)

            if cfg.patience and stalled >= cfg.patience:
                reason = f"early stop (no gain in {cfg.patience} epochs)"
                break
    except KeyboardInterrupt:
        reason = "interrupted"

    run.log(f"{reason}. best val IoU {best_iou:.4f} -> {run.checkpoint('best')}")
    run.close()
    if wb is not None:
        wb.summary["best_iou"] = best_iou
        wb.summary["stop_reason"] = reason
        wb.finish()


if __name__ == "__main__":
    main()
