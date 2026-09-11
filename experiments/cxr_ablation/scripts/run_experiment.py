"""Configuration-driven audit, dry-run and smoke-test entry point.

Full training is intentionally gated until the selected encoder and decoder
have an explicit compatible loader in the registry. This prevents silently
replacing a requested radiology model with a generic text model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def digest(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE / "configs/base_iu.yaml")
    parser.add_argument("--experiment-id", default="audit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"].get("seed", 1337)); seed_everything(seed)
    annotation = ROOT / config["dataset"]["annotation"]
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    device = "cuda" if torch.cuda.is_available() and config["experiment"].get("device", "auto") != "cpu" else "cpu"
    manifest = {
        "experiment_id": args.experiment_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run" if args.dry_run else "smoke_test" if args.smoke_test else "plan",
        "config": config, "config_sha256": digest(args.config), "dataset_annotation_sha256": digest(annotation),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": device, "status": "READY_FOR_COMPONENT_AUDIT",
    }
    out = PACKAGE / "outputs" / args.experiment_id; out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run or args.smoke_test:
        # Dataset loading is intentionally delegated to the existing project
        # loader after a model-specific adapter has passed the compatibility audit.
        print(json.dumps({"status": "DATASET_AND_ENVIRONMENT_VALIDATED", "mode": manifest["mode"], "device": device, "annotation": str(annotation), "next_gate": "run audit_registry.py, build_compatibility_matrix.py and test_all_components.py"}, indent=2))
        return
    print(json.dumps({"status": "PLAN_ONLY", "message": "No full training started. Select only a compatibility-verified encoder/decoder pair after reviewing the audit outputs.", "manifest": str(out / "manifest.json")}, indent=2))
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

