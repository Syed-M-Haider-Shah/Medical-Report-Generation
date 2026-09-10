"""Decoder registry with explicit compatibility boundaries.

Qwen and Llama are declared for future ablations, but are intentionally not
constructed until a tested visual-token adapter and local checkpoint are added.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DecoderSpec:
    name: str
    hidden_size: int | None
    visual_interface: str
    local_path: str
    checkpoint_policy: str


DECODERS = {
    "distilgpt2_iu": DecoderSpec("distilgpt2_iu", 768, "encoder_outputs.last_hidden_state", "models/distilgpt2", "original or explicitly named fine-tuned checkpoint"),
    "qwen2": DecoderSpec("qwen2", None, "visual-token adapter required", "models/qwen2", "local checkpoint required"),
    "llama": DecoderSpec("llama", None, "visual-token adapter required", "models/llama", "local checkpoint required"),
}


def describe() -> dict[str, dict[str, Any]]:
    return {name: asdict(spec) for name, spec in DECODERS.items()}


def build(name: str, config: Any, device: Any, checkpoint: str | None = None):
    if name not in DECODERS:
        raise KeyError(f"Unknown decoder {name!r}; choose {sorted(DECODERS)}")
    if name != "distilgpt2_iu":
        raise NotImplementedError(
            f"{name} is registry-only: declare and test a visual-token adapter before running this ablation"
        )
    raise NotImplementedError(
        "Use the existing CvT2DistilGPT2 loader for distilgpt2_iu; this registry does not duplicate checkpoint loading"
    )

