"""Versioned vision-encoder registry for the ablation study.

This module describes interfaces and checkpoint provenance. It never downloads
weights implicitly; every experiment must point to a local, recorded checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VisionEncoderSpec:
    name: str
    feature_dim: int
    token_layout: str
    input_size: int
    checkpoint: str
    frozen_by_default: bool = True


VISION_ENCODERS = {
    "cvt21_original": VisionEncoderSpec(
        "cvt21_original", 384, "[B,N,D] after flatten", 384,
        "models/cvt2distilgpt2-iu", True,
    ),
    "biovilt": VisionEncoderSpec(
        "biovilt", 512, "[B,14,14,D] raw patches", 448,
        "models/biovil-t/biovil_t_image_model_proj_size_128.pt", True,
    ),
}


def describe() -> dict[str, dict[str, Any]]:
    return {name: asdict(spec) for name, spec in VISION_ENCODERS.items()}


def build(name: str, config: Any, device: Any):
    """Build a frozen adapter using the model's official implementation.

    Imports are lazy so registry inspection and CSV collection do not require a
    GPU environment. New encoders must be added here explicitly and tested.
    """
    import torch
    from torch import nn

    if name not in VISION_ENCODERS:
        raise KeyError(f"Unknown vision encoder {name!r}; choose {sorted(VISION_ENCODERS)}")
    spec = VISION_ENCODERS[name]
    if name == "biovilt":
        from health_multimodal.image.model.model import ImageEncoderType, ImageModel

        checkpoint = ROOT / spec.checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"BioViL-T checkpoint not found: {checkpoint}")
        model = ImageModel.from_pretrained(str(checkpoint), ImageEncoderType.RESNET50_MULTI_IMAGE)
    elif name == "cvt21_original":
        import sys
        import os
        sys.path.insert(0, str(ROOT / "vendor/cvt2distilgpt2"))
        from tools.cvt import CvT
        model = CvT(config).to(device)
        checkpoint = ROOT / spec.checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"CvT checkpoint directory not found: {checkpoint}")
        raise NotImplementedError("Use the official CvT loader in the existing pipeline for this checkpoint")
    else:  # pragma: no cover
        raise AssertionError(name)

    class Adapter(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped.eval()
            for parameter in self.wrapped.parameters():
                parameter.requires_grad_(False)

        def forward(self, images):
            with torch.no_grad():
                output = self.wrapped(images)
                if isinstance(output, dict):
                    output = output.get("patch_embeddings", output.get("last_hidden_state"))
                if output is None:
                    raise ValueError("Vision encoder did not return patch embeddings")
                if output.ndim == 4:
                    output = output.flatten(2).transpose(1, 2)
                if output.ndim != 3 or output.shape[-1] != spec.feature_dim:
                    raise ValueError(f"{name} returned shape {tuple(output.shape)}, expected [B,N,{spec.feature_dim}]")
                return output

    return Adapter(model.to(device)), spec

