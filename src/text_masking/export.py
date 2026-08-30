"""Export a labelled `.cvlabel` to a YOLO detection dataset.

    Speech Bubble / Bubble  -> 0 (text_bubble)
    Text on Page / Text On Page -> 1 (text_free)

Output:

    <out>/data.yaml
    <out>/{train,valid,test}/{images,labels}/NNNNNN.{png,txt}
"""

import os

import cv2

from . import cvlabel

CLASS_ID = {
    "Speech Bubble": 0, "Bubble": 0,
    "Text on Page": 1, "Text On Page": 1,
}
NAMES = ["text_bubble", "text_free"]
YOLO_SPLIT = {"train": "train", "valid": "val", "test": "test"}


def export_detect(labels: str, out: str) -> None:
    samples = cvlabel.read(labels)
    splits = sorted({s.split for s in samples})
    for split in splits:
        os.makedirs(os.path.join(out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, split, "labels"), exist_ok=True)

    n_box = 0
    for i, s in enumerate(samples):
        h, w = s.image.shape[:2]
        stem = f"{i:06d}"
        cv2.imwrite(
            os.path.join(out, s.split, "images", stem + ".png"),
            cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR),
        )
        lines = []
        for a in s.annotations:
            cid = CLASS_ID.get(a.label)
            if cid is None:
                continue
            x1, y1, x2, y2 = a.xyxy
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(
                f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
            n_box += 1
        with open(os.path.join(out, s.split, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out)}\n")
        for split in splits:
            f.write(f"{YOLO_SPLIT.get(split, split)}: {split}/images\n")
        f.write(f"\nnc: {len(NAMES)}\nnames: {NAMES}\n")

    print(f"{len(samples)} images, {n_box} boxes -> {out}  splits={splits}")
