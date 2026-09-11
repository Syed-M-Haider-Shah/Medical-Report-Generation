"""Build an explicit encoder/decoder compatibility matrix from the registry."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
REGISTRY = PACKAGE / "configs/model_registry.yaml"
OUT = PACKAGE / "outputs/component_compatibility_matrix.csv"


def classification(encoder: dict[str, Any], decoder: dict[str, Any]) -> tuple[str, str]:
    if decoder.get("role") == "complete_external_baseline":
        return "FULL_MODEL_ONLY", "Decoder is registered as a complete pretrained system"
    if encoder.get("component_compatible") is False or decoder.get("component_compatible") is False:
        return "FULL_MODEL_ONLY", encoder.get("compatibility_reason") or decoder.get("compatibility_reason") or "Native multimodal connector must be preserved"
    if decoder.get("supports_external_inputs_embeds") is True:
        return "PROJECTOR_COMPATIBLE", "External visual embeddings are explicitly supported"
    return "ADAPTER_EXTRACTION_REQUIRED", "External embedding interface is not yet verified"


def main() -> None:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    rows = []
    for ename, encoder in data.get("encoders", {}).items():
        row = {"encoder": ename}
        for dname, decoder in data.get("decoders", {}).items():
            value, reason = classification(encoder, decoder)
            row[dname] = value
            rows.append({"encoder": ename, "decoder": dname, "classification": value, "reason": reason})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["encoder", "decoder", "classification", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "WRITTEN", "cells": len(rows), "path": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

