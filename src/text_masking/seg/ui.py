"""Gradio UI to test a trained checkpoint on uploaded images.

    tm-ui --checkpoint runs/<run>/checkpoints/best.pt
"""

import os
from dataclasses import dataclass

import cv2
import gradio as gr
import numpy as np
import torch

from ..args import parse
from .infer import load_model, mask_from_prob, predict_prob


@dataclass
class Config:
    checkpoint: str = "./runs/latest/checkpoints/best.pt"
    device: str = "cuda:0"
    port: int = 7860
    share: bool = False  # create a public gradio link


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    hit = mask > 0
    out[hit] = (0.45 * rgb[hit] + np.array([0.55 * 255, 0, 0])).astype(np.uint8)
    return out


def heatmap(prob: np.ndarray) -> np.ndarray:
    """blue = confident background, green/yellow = unsure, red = confident text."""
    bgr = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_ui(model, imgsz: int, device: torch.device, metrics: dict) -> gr.Blocks:
    caption = f"checkpoint imgsz={imgsz}"
    if metrics.get("iou"):
        caption += (
            f"  ·  val IoU {metrics['iou']:.3f}"
            f"  P {metrics['precision']:.3f}  R {metrics['recall']:.3f}"
        )

    def run(image, threshold, dilate):
        if image is None:
            return None, None, None
        prob = predict_prob(model, imgsz, device, image)
        mask = mask_from_prob(prob, threshold, dilate)
        return overlay(image, mask), mask, heatmap(prob)

    with gr.Blocks(title="Text-mask segmenter") as demo:
        gr.Markdown(f"## Text-mask segmenter\n{caption}")
        with gr.Row():
            image = gr.Image(type="numpy", label="Upload", height=380)
            with gr.Column():
                threshold = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="threshold")
                dilate = gr.Slider(0, 12, 0, step=1, label="dilate mask (px)")
        with gr.Row():
            out_overlay = gr.Image(label="overlay")
            out_mask = gr.Image(label="mask")
            out_prob = gr.Image(label="probability (blue=bg · green=unsure · red=text)")

        inputs = [image, threshold, dilate]
        outputs = [out_overlay, out_mask, out_prob]
        for control in inputs:
            control.change(run, inputs, outputs)

    return demo


def main() -> None:
    cfg = parse(Config)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available -> CPU.")
        cfg.device = "cpu"
    device = torch.device(cfg.device)

    ckpt = cfg.checkpoint
    if not os.path.exists(ckpt) and "latest" in ckpt:
        runs = sorted(d for d in os.listdir("./runs") if os.path.isdir(f"./runs/{d}"))
        if runs:
            ckpt = f"./runs/{runs[-1]}/checkpoints/best.pt"

    model, imgsz, metrics = load_model(ckpt, device)
    print(f"loaded {ckpt}  (imgsz {imgsz}, {device})")
    build_ui(model, imgsz, device, metrics).launch(
        server_port=cfg.port, share=cfg.share
    )


if __name__ == "__main__":
    main()
