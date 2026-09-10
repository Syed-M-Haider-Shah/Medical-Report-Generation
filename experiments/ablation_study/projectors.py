"""Explicit projector registry for vision-token to decoder-token adapters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectorSpec:
    name: str
    kind: str
    output_dim: int
    trainable_parameters: str


PROJECTORS = {
    "linear": ProjectorSpec("linear", "Linear", 768, "weight,bias"),
    "mlp_gelu_ln": ProjectorSpec("mlp_gelu_ln", "Linear-GELU-Linear-LayerNorm", 768, "all projector parameters"),
    "identity": ProjectorSpec("identity", "Identity", 0, "none"),
}


def describe() -> dict[str, dict[str, Any]]:
    return {name: asdict(spec) for name, spec in PROJECTORS.items()}


def build(name: str, input_dim: int, output_dim: int = 768):
    import torch.nn as nn

    if name not in PROJECTORS:
        raise KeyError(f"Unknown projector {name!r}; choose {sorted(PROJECTORS)}")
    if name == "linear":
        return nn.Linear(input_dim, output_dim)
    if name == "mlp_gelu_ln":
        hidden = max(output_dim, 2 * input_dim)
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim), nn.LayerNorm(output_dim))
    if input_dim != output_dim:
        raise ValueError("identity projector requires input_dim == output_dim")
    return nn.Identity()


def parameter_report(module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}

