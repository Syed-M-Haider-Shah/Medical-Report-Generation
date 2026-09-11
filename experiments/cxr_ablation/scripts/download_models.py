"""Resumable, registry-driven model downloads.

Only official Hugging Face repositories in the registry are downloaded. Models
without a verified repository are recorded as unavailable and never guessed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
REGISTRY = PACKAGE / "configs/model_registry.yaml"
OUT = PACKAGE / "outputs/download_status.json"
CACHE = ROOT / "cache/models"


def flatten(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: {"category": section.rstrip("s"), **spec} for section, models in data.items() for name, spec in models.items()}


def selected(args, registry):
    models = flatten(registry)
    if args.model:
        names = [args.model]
    elif args.encoders:
        names = [n for n, s in models.items() if s["category"] == "encoder"]
    elif args.decoders:
        names = [n for n, s in models.items() if s["category"] == "decoder"]
    elif args.full_models:
        names = [n for n, s in models.items() if s["category"] == "full_model"]
    else:
        names = list(models) if args.all else []
    unknown = [n for n in names if n not in models]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}")
    return [(name, models[name]) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--encoders", action="store_true")
    group.add_argument("--decoders", action="store_true")
    group.add_argument("--full-models", action="store_true")
    group.add_argument("--model")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    prior = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    results = dict(prior)
    for name, spec in selected(args, registry):
        target = CACHE / name
        record = {"model_name": name, "category": spec["category"], "started_at": datetime.now(timezone.utc).isoformat(), "local_path": str(target)}
        try:
            if not spec.get("huggingface_repo"):
                record.update(status="NOT_ATTEMPTED", reason="No verified official Hugging Face repository in registry; manual source verification required")
            elif target.exists() and any(target.rglob("*")) and not args.force:
                record.update(status="ALREADY_PRESENT", repository=spec["huggingface_repo"])
            else:
                from huggingface_hub import snapshot_download
                token = os.environ.get("HF_TOKEN")
                snapshot_download(repo_id=spec["huggingface_repo"], local_dir=str(target), local_dir_use_symlinks=False, token=token, resume_download=True)
                record.update(status="DOWNLOADED", repository=spec["huggingface_repo"])
        except Exception as exc:  # continue to other models by design
            record.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=3))
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        results[name] = record
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))
    print(json.dumps({"status_file": str(OUT), "processed": len(selected(args, registry))}, indent=2))


if __name__ == "__main__":
    main()

