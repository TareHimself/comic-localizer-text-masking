"""Read / write ``.cvlabel`` archives -- the native format of cv-label
(https://github.com/TareHimself/cv-label).

A ``.cvlabel`` is a zip holding ``manifest.json`` + ``images/<id>.<ext>``.
Annotations are ``box`` (2 opposite corners) or ``polygon`` (3+ vertices), in
absolute pixel coordinates. Format spec:
https://github.com/TareHimself/cv-label/blob/main/formats/cvlabel/SPEC.md

Manifest ``version`` is ``1``; earlier, unversioned manifests are not read.
"""

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

VERSION = 1


@dataclass
class Annotation:
    label: str  # resolved label name
    kind: str  # "box" | "polygon"
    points: list[tuple[float, float]]

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def polygon(self) -> np.ndarray:
        """Outline as an int32 (N, 2) array. A box becomes its 4 corners."""
        if self.kind == "box":
            x1, y1, x2, y2 = self.xyxy
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        return np.array(self.points, np.int32)


@dataclass
class Sample:
    name: str
    image: np.ndarray  # RGB uint8
    annotations: list[Annotation] = field(default_factory=list)
    split: str = "train"
    task: str = ""  # originating task name; write() groups samples back by it

    def by_label(self, *labels: str) -> list[Annotation]:
        want = set(labels)
        return [a for a in self.annotations if a.label in want]


def _ordered_points(a: dict) -> list[tuple[float, float]]:
    pts = a["points"]
    if all("sequence" in p for p in pts):
        pts = sorted(pts, key=lambda p: p["sequence"])
    return [(p["x"], p["y"]) for p in pts]


def read(path: str) -> list[Sample]:
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        name_of = {lb["id"]: lb["name"] for lb in manifest["labels"]}
        samples = []
        for task in manifest["tasks"]:
            for s in task["samples"]:
                buf = np.frombuffer(z.read(s["imageFile"]), np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                samples.append(
                    Sample(
                        name=s["name"],
                        image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                        split=s.get("split", "train"),
                        task=task["name"],
                        annotations=[
                            Annotation(
                                label=name_of.get(a["labelId"], a["labelId"]),
                                kind=a["type"],
                                points=_ordered_points(a),
                            )
                            for a in s["annotations"]
                        ],
                    )
                )
    return samples


def _color(name: str) -> str:
    return "#" + hashlib.md5(name.encode()).hexdigest()[:6]


def write(path: str, samples: list[Sample]) -> None:
    names = sorted({a.label for s in samples for a in s.annotations})
    label_id = {name: f"lbl_{i}" for i, name in enumerate(names)}
    now = datetime.now(timezone.utc).isoformat()

    tasks: dict[str, list[Sample]] = {}
    for s in samples:
        tasks.setdefault(s.task or Path(path).stem, []).append(s)

    manifest = {
        "version": VERSION,
        "labels": [
            {"id": lid, "name": n, "color": _color(n)} for n, lid in label_id.items()
        ],
        "tasks": [],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        i = 0
        for t, (task_name, task_samples) in enumerate(tasks.items()):
            task = {"id": f"task_{t}", "name": task_name, "samples": []}
            for s in task_samples:
                sid = f"smp_{i}"
                i += 1
                fname = f"images/{sid}.png"
                _, png = cv2.imencode(".png", cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR))
                z.writestr(fname, png.tobytes())
                h, w = s.image.shape[:2]
                task["samples"].append(
                    {
                        "id": sid,
                        "name": s.name,
                        "split": s.split,
                        "createdAt": now,
                        "completedAt": None,
                        "imageFile": fname,
                        "width": w,
                        "height": h,
                        "annotations": [
                            {
                                "id": f"{sid}_a{j}",
                                "type": a.kind,
                                "labelId": label_id[a.label],
                                "points": [
                                    {
                                        "id": f"{sid}_a{j}_p{k}",
                                        "x": float(x),
                                        "y": float(y),
                                        "sequence": k,
                                    }
                                    for k, (x, y) in enumerate(a.points)
                                ],
                            }
                            for j, a in enumerate(s.annotations)
                        ],
                    }
                )
            manifest["tasks"].append(task)
        z.writestr("manifest.json", json.dumps(manifest))
