"""Run non-destructive local component sanity checks.

The audit records missing dependencies/checkpoints instead of substituting a
generic model or claiming a successful feature extraction.
"""
from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
OUT = PACKAGE / "outputs/encoder_feature_audit.csv"


def main() -> None:
    registry = __import__("yaml").safe_load((PACKAGE / "configs/model_registry.yaml").read_text(encoding="utf-8"))
    rows = []
    for name, spec in registry.get("encoders", {}).items():
        local = ROOT / "cache/models" / name
        row = {"model_name": name, "device": "not_run", "input_shape": "", "resolution": "", "raw_output_type": "", "raw_feature_shape": "", "selected_feature_shape": "", "visual_tokens": "", "embedding_dim": "", "dtype": "", "inference_seconds": "", "representation_type": "", "status": "NOT_TESTED", "reason": ""}
        if not local.exists():
            row.update(status="NOT_AVAILABLE", reason=f"Local checkpoint directory absent: {local}")
        else:
            row.update(status="CHECKPOINT_PRESENT_NOT_LOADED", reason="Model-specific official processor/loader must be registered before extraction")
        rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "WRITTEN", "models": len(rows), "path": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

