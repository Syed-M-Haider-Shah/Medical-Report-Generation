"""Aggregate completed experiment metric JSON files into a reproducible CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=PACKAGE / "outputs")
    parser.add_argument("--output", type=Path, default=PACKAGE / "outputs/results_summary.csv")
    args = parser.parse_args()
    records = []
    for path in sorted(args.metrics_root.glob("**/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8")); metrics = data.get("metrics", data)
        records.append({"experiment_id": path.parent.name, "metrics_path": str(path.relative_to(PACKAGE)), **{str(k): v for k, v in metrics.items() if isinstance(v, (int, float, str))}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record}) if records else ["experiment_id", "metrics_path"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    print(json.dumps({"status": "WRITTEN", "experiments": len(records), "path": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

