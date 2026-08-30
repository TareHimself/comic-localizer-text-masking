"""A run directory: `runs/<timestamp>_<name>/` holding everything one training
run produces -- config, per-epoch metrics, checkpoints, sample overlays, log.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class RunDir:
    def __init__(self, root: str, name: str = ""):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(root) / (f"{stamp}_{name}" if name else stamp)
        (self.path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.path / "samples").mkdir(exist_ok=True)
        self._metrics = self.path / "metrics.csv"
        self._log = (self.path / "train.log").open("a", encoding="utf-8")

    def save_config(self, cfg: dict) -> None:
        (self.path / "config.json").write_text(
            json.dumps(cfg, indent=2, default=str), encoding="utf-8"
        )

    def log(self, msg: str) -> None:
        print(msg)
        self._log.write(msg + "\n")
        self._log.flush()

    def log_metrics(self, row: dict) -> None:
        header = not self._metrics.exists()
        with self._metrics.open("a", encoding="utf-8") as f:
            if header:
                f.write(",".join(row) + "\n")
            f.write(
                ",".join(
                    f"{v:.6g}" if isinstance(v, float) else str(v)
                    for v in row.values()
                )
                + "\n"
            )

    def checkpoint(self, name: str) -> Path:
        return self.path / "checkpoints" / f"{name}.pt"

    def save_overlays(self, epoch: int, panels: list[np.ndarray]) -> None:
        """`panels` are HxWx3 BGR images (image | pred | gt strips); stacked."""
        if not panels:
            return
        grid = np.vstack(
            [cv2.resize(p, (panels[0].shape[1], panels[0].shape[0])) for p in panels]
        )
        cv2.imwrite(str(self.path / "samples" / f"epoch_{epoch:03d}.png"), grid)

    def close(self) -> None:
        self._log.close()
