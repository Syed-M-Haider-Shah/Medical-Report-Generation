"""Central validation and results collector for model ablations.

Every completed full experiment is represented by one CSV row. Model component
interfaces live in the three sibling registries: vision_encoders.py,
projectors.py and llm_decoders.py. This file owns split validation, provenance
and metric aggregation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/ablation_study/outputs"

METRICS = [
    "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "ROUGE_L", "ROUGE_1", "ROUGE_2",
    "ROUGE_L_F1", "METEOR", "CIDEr", "BERTScore", "CheXbert_14_micro_F1",
    "CheXbert_14_macro_F1", "CheXbert_5_micro_F1", "RadGraph", "GREEN",
]

# Paths are deliberately explicit. Add future complete runs here; do not make
# another collector script with a different metric or split convention.
EXPERIMENTS = {
    "cvt2distilgpt2_official": {
        "feature_extractor": "CvT-21-384x384-IN-22k",
        "decoder": "DistilGPT2 IU checkpoint",
        "projector": "Linear 384->768 (original)",
        "training_regime": "original pretrained checkpoint",
        "metrics": "experiments/biovilt2distilgpt2/outputs/comparison/metrics.json",
        "predictions": "experiments/biovilt2distilgpt2/outputs/comparison/official_cvt2distilgpt2_test_predictions.csv",
        "checkpoint": "models/cvt2distilgpt2-iu/epoch=10-val_chen_cider=0.475024.ckpt",
    },
    "biovilt_distilgpt2_b1": {
        "feature_extractor": "Microsoft BioViL-T (frozen)",
        "decoder": "DistilGPT2 IU checkpoint (frozen)",
        "projector": "Linear 512->768",
        "training_regime": "projector-only B1",
        "metrics": "experiments/biovilt2distilgpt2/outputs/full/B1/metrics.json",
        "predictions": "experiments/biovilt2distilgpt2/outputs/full/B1/test_predictions.csv",
        "checkpoint": "models/biovilt2distilgpt2/full/biovilt_projector_stage1_best.pt",
    },
    "biovilt_distilgpt2_b2": {
        "feature_extractor": "Microsoft BioViL-T (frozen)",
        "decoder": "DistilGPT2 IU checkpoint (fine-tuned)",
        "projector": "Linear 512->768",
        "training_regime": "projector + decoder B2",
        "metrics": "experiments/biovilt2distilgpt2/outputs/full/B2/metrics.json",
        "predictions": "experiments/biovilt2distilgpt2/outputs/full/B2/test_predictions.csv",
        "checkpoint": "models/biovilt2distilgpt2/full/biovilt2distilgpt2_best.pt",
    },
    "biovilt_distilgpt2_clinical": {
        "feature_extractor": "Microsoft BioViL-T (frozen)",
        "decoder": "DistilGPT2 B2 decoder (fine-tuned)",
        "projector": "MLP 512->1024->768 + clinical head",
        "training_regime": "report CE + CheXbert auxiliary loss",
        "metrics": "experiments/biovilt2distilgpt2/outputs/clinical/metrics.json",
        "predictions": "experiments/biovilt2distilgpt2/outputs/clinical/test_predictions.csv",
        "checkpoint": "models/biovilt2distilgpt2-clinical/best.pt",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str) -> dict:
    full = ROOT / path
    if not full.is_file():
        raise FileNotFoundError(full)
    return json.loads(full.read_text(encoding="utf-8"))


def prediction_ids(path: str) -> tuple[int, list[str], str]:
    full = ROOT / path
    if not full.is_file():
        raise FileNotFoundError(full)
    with full.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["study_id"] for row in rows]
    if len(rows) != 590 or len(set(ids)) != 590:
        raise ValueError(f"{path}: expected 590 unique test studies, found {len(rows)} rows")
    return len(rows), ids, sha256(full)


def collect() -> list[dict]:
    rows, expected_ids = [], None
    for experiment_id, spec in EXPERIMENTS.items():
        metrics = load(spec["metrics"])
        count, ids, prediction_hash = prediction_ids(spec["predictions"])
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"{experiment_id}: test study IDs/order differ from the first experiment")
        row = {"experiment_id": experiment_id, "split": "IU R2Gen test", "studies": count,
               "feature_extractor": spec["feature_extractor"], "decoder": spec["decoder"],
               "projector": spec["projector"], "training_regime": spec["training_regime"],
               "checkpoint": spec["checkpoint"], "prediction_sha256": prediction_hash}
        row.update({metric: metrics.get("metrics", {}).get(metric, "") for metric in METRICS})
        row["metric_availability"] = json.dumps(metrics.get("availability", {}), ensure_ascii=False, sort_keys=True)
        rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ablation_results.csv"
    fields = ["experiment_id", "split", "studies", "feature_extractor", "decoder", "projector",
              "training_regime", "checkpoint", "prediction_sha256"] + METRICS + ["metric_availability"]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect", "show", "validate", "registry"), default="collect")
    args = parser.parse_args()
    if args.mode == "registry":
        # Lazy imports keep validation usable from a lightweight Python install.
        from vision_encoders import describe as describe_vision
        from projectors import describe as describe_projectors
        from llm_decoders import describe as describe_decoders
        print(json.dumps({"vision_encoders": describe_vision(),
                          "projectors": describe_projectors(),
                          "llm_decoders": describe_decoders()}, indent=2))
        return
    rows = collect()
    path = write_csv(rows)
    if args.mode == "show":
        for row in rows:
            print(json.dumps({k: row[k] for k in ("experiment_id", *METRICS)}, indent=2))
    elif args.mode == "validate":
        print(json.dumps({"status": "VALID", "experiments": len(rows), "same_test_order": True,
                          "studies_per_experiment": 590, "csv": str(path)}, indent=2))
    else:
        print(json.dumps({"status": "COLLECTED", "experiments": len(rows), "studies_per_experiment": 590,
                          "csv": str(path)}, indent=2))


if __name__ == "__main__":
    main()

