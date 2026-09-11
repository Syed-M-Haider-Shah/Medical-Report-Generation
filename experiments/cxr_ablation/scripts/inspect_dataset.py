"""Inspect the existing IU R2Gen annotation without changing it or creating splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
OUT = PACKAGE / "outputs/dataset_audit.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=ROOT / "datasets/iu_xray/annotation.json")
    args = parser.parse_args()
    annotation = args.annotation if args.annotation.is_absolute() else ROOT / args.annotation
    if not annotation.is_file():
        raise FileNotFoundError(f"Annotation not found: {annotation}")
    data = json.loads(annotation.read_text(encoding="utf-8"))
    try:
        annotation_display = str(annotation.relative_to(ROOT))
    except ValueError:
        annotation_display = str(annotation)
    result = {"annotation": annotation_display, "sha256": digest(annotation), "splits": {}}
    all_ids = []
    for split in ("train", "val", "test"):
        entries = data.get(split, [])
        ids = [str(item.get("id", item.get("study_id", ""))) for item in entries]
        paths = [p for item in entries for p in item.get("image_path", item.get("image_paths", []))]
        result["splits"][split] = {"studies": len(entries), "unique_study_ids": len(set(ids)), "image_references": len(paths), "duplicate_image_references": len(paths) - len(set(paths)), "views_per_study": dict(Counter(len(item.get("image_path", item.get("image_paths", []))) for item in entries))}
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Study leakage detected: an ID appears in multiple splits")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

