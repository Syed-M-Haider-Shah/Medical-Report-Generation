"""Generate the pre-download model availability and provenance reports."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
REGISTRY = PACKAGE / "configs/model_registry.yaml"
OUT = PACKAGE / "outputs"


def flatten(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for kind, models in registry.items():
        for name, spec in models.items():
            rows.append({"model_name": name, "category": kind.rstrip("s"), **spec})
    return rows


def local_state(name: str) -> tuple[str, str]:
    path = ROOT / "cache/models" / name
    if path.exists() and any(path.rglob("*")):
        return "LOCAL_FILES_PRESENT", f"cache/models/{name}"
    return "NOT_DOWNLOADED", f"cache/models/{name}"


def main() -> None:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    rows = []
    for item in flatten(registry):
        status, local_path = local_state(item["model_name"])
        rows.append({
            **item,
            "authors_or_source": item.get("official_repo_url") or "NOT VERIFIED",
            "downloadable": "YES" if item.get("huggingface_repo") or item.get("official_repo_url") else "UNKNOWN",
            "authentication_required": "UNKNOWN; check model card" if item.get("huggingface_repo") else "UNKNOWN",
            "license_status": item.get("license", "UNVERIFIED"),
            "parameter_count": "NOT RECORDED UNTIL CONFIG IS LOADED",
            "input_format": "MODEL-SPECIFIC; verify official processor",
            "resolution": "MODEL-SPECIFIC; verify official processor",
            "visual_representation": "NOT RECORDED UNTIL SANITY TEST",
            "patch_tokens": "UNKNOWN",
            "download_status": status,
            "local_checkpoint_path": local_path,
            "failure_reason": "" if status == "LOCAL_FILES_PRESENT" else "No download attempted or official source is unverified",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (OUT / "model_availability_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Model availability audit", "", "Generated before model downloads. Unknown fields are explicit and are not inferred.", ""]
    lines.append(f"Environment: Python {platform.python_version()}, platform {platform.platform()}")
    lines.append("")
    for row in rows:
        lines += [f"## {row['model_name']}", f"- Category: {row['category']}", f"- Official source: {row['official_repo_url'] or 'NOT VERIFIED'}", f"- Hugging Face: {row['huggingface_repo'] or 'NONE RECORDED'}", f"- CXR/report training claim: {row['trained_on']}", f"- Licence: {row['license']}", f"- Download status: {row['download_status']}", f"- Component-compatible: {row.get('component_compatible', 'UNKNOWN')}", f"- Local path: {row['local_checkpoint_path']}", ""]
    (OUT / "model_availability_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "environment_audit.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata.get("Name") in {"torch", "transformers", "timm", "huggingface-hub", "torchvision", "PyYAML"}}}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "WRITTEN", "models": len(rows), "csv": str(OUT / "model_availability_report.csv"), "markdown": str(OUT / "model_availability_report.md")}, indent=2))


if __name__ == "__main__":
    main()

