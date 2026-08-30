"""Model + loss for binary text-mask segmentation."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

ARCHS = {"unet": smp.Unet, "fpn": smp.FPN}


def build_model(arch: str, encoder: str, pretrained: bool = True) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {list(ARCHS)}")
    return ARCHS[arch](
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
    )


def dice_bce_loss():
    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()

    def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice(logits, target) + bce(logits, target)

    return loss_fn
