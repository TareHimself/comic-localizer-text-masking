"""Split dataset generation across processes.

A `Producer` yields `(Sample, [box], rng)` triples; `run` slices the producer's
item range across worker processes and writes each page whole or as crops.
Output stems are `<shard>_<n>` so shards never collide. The exact dataset is
only reproducible for a fixed `(seed, workers)`.
"""

import json
import multiprocessing as mp
import os
import queue as _queue
import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import cv2
from tqdm import tqdm

from .crops import page_crops
from .render import Box, Sample

Page = tuple[Sample, list[Box], random.Random]


class Producer(Protocol):
    def produce(self, shard: int, lo: int, hi: int) -> Iterator[Page]: ...


def _dirs(out_dir: str) -> tuple[str, str]:
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    return img_dir, mask_dir


def _save(img_dir: str, mask_dir: str, stem: str, sample: Sample) -> None:
    cv2.imwrite(
        os.path.join(img_dir, stem + ".png"),
        cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(mask_dir, stem + ".png"), sample.mask)


@dataclass
class _Shard:
    producer: Producer
    out_dir: str
    index: int
    lo: int
    hi: int
    crops: bool


def _run_shard(shard: _Shard, progress=None) -> int:
    img_dir, mask_dir = _dirs(shard.out_dir)
    written = pending = 0
    for page, boxes, rng in shard.producer.produce(shard.index, shard.lo, shard.hi):
        samples = page_crops(page, boxes, rng) if shard.crops else (page,)
        for sample in samples:
            _save(img_dir, mask_dir, f"{shard.index:02d}_{written:07d}", sample)
            written += 1
            pending += 1
            if progress is not None and pending >= 50:
                progress.put(pending)
                pending = 0
    if progress is not None and pending:
        progress.put(pending)
    return written


def _worker_count(workers: int) -> int:
    return min(os.cpu_count() or 4, 8) if workers <= 0 else workers


def _splits(n: int, parts: int) -> list[int]:
    return [round(n * k / parts) for k in range(parts + 1)]


def run(
    producer: Producer,
    out_dir: str,
    *,
    n_items: int,
    total: int,
    crops: bool,
    workers: int,
    desc: str,
    manifest: dict | None = None,
) -> None:
    """Generate into `out_dir`. If `manifest` is given, it is written to
    `out_dir/manifest.json` with the actual `written` count added -- this is the
    provenance record `tm-pack` and `tm-train` read back."""
    n = _worker_count(workers)
    _dirs(out_dir)
    edges = _splits(n_items, n)
    shards = [
        _Shard(producer, out_dir, k, edges[k], edges[k + 1], crops)
        for k in range(n)
        if edges[k + 1] > edges[k]
    ]
    written = 0
    if len(shards) <= 1:
        with tqdm(total=total or None, desc=desc) as bar:
            for s in shards:
                w = _run_shard(s)
                written += w
                bar.update(w)
    else:
        with mp.Manager() as mgr, ProcessPoolExecutor(max_workers=len(shards)) as ex:
            q = mgr.Queue()
            futures = [ex.submit(_run_shard, s, q) for s in shards]
            with tqdm(total=total or None, desc=desc) as bar:
                while any(not f.done() for f in futures):
                    try:
                        bar.update(q.get(timeout=0.3))
                    except _queue.Empty:
                        pass
                while True:  # final drain of counts queued as workers exited
                    try:
                        bar.update(q.get_nowait())
                    except _queue.Empty:
                        break
            written = sum(f.result() for f in futures)  # also surfaces exceptions

    if manifest is not None:
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({**manifest, "written": written}, f, indent=2)
