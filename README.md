# IU X-Ray evaluation with Janus-Pro-CXR-Zero

This repository evaluates the frozen local `ZrH42/Janus-Pro-CXR-Zero` checkpoint on study-level IU X-Ray/Open-I cases. It performs no training.

## Split terminology

The raw NLM/Open-I release contains images and XML reports but no publisher-defined train/validation/test split. This project uses the established R2Gen benchmark `annotation.json` split and calls it the **R2Gen benchmark test split**. It never invents or randomly regenerates a test split.

## Expected data

```text
datasets/iu_xray/
â”œâ”€â”€ annotation.json
â””â”€â”€ images/
    â””â”€â”€ *.png
```

The annotation must have R2Gen's structure:

```json
{
  "train": [],
  "val": [],
  "test": [
    {"id": "CXR...", "image_path": ["...png", "...png"], "report": "..."}
  ]
}
```

Every test entry remains one study, even when it contains frontal and lateral images.
The downloaded R2Gen annotation uses renamed paths such as `study/0.png`; the
loader deterministically resolves these to the original NLM filenames. R2Gen's
two single-image studies repeat their sole image in both annotation slots.

## Environment

Use Python 3.10. The currently detected system Python 3.14 is outside the authors' tested dependency range.

```powershell
conda create -n cxr-agent python=3.10 -y
conda activate cxr-agent
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

New-Item -ItemType Directory -Force vendor | Out-Null
git clone https://github.com/ZrH42/Janus-Pro-CXR.git vendor/Janus-Pro-CXR
pip install -e vendor/Janus-Pro-CXR
```

The model is loaded offline from `models/janus-pro-cxr-zero`. The official Janus implementation is used for processing and generation.

## Validate and run

First ensure the dataset finished downloading and that `datasets/iu_xray/annotation.json` is present. Then run one study:

```powershell
python scripts/evaluate_iu_xray.py --config config.yaml --validate-only
```

This validates every test entry and referenced image without loading the model. Then run one study:

```powershell
python scripts/evaluate_iu_xray.py --config config.yaml --limit 1
```

Run a five-study smoke test:

```powershell
python scripts/evaluate_iu_xray.py --config config.yaml --limit 5
```

Run the complete R2Gen benchmark test split:

```powershell
python scripts/evaluate_iu_xray.py --config config.yaml
```

Recalculate metrics without loading Janus or regenerating reports:

```powershell
python scripts/evaluate_iu_xray.py --config config.yaml --metrics-only
```

Predictions are appended to `outputs/iu_xray/direct/predictions.jsonl` after every successful case, so an interrupted run resumes from its cache. Use `--overwrite` only when intentionally replacing the cached experiment.

The current evaluator reports BLEU-1 through BLEU-4, ROUGE-1/2/L, and METEOR. Clinical metrics and the agentic refinement experiment should be added after one real-image Janus smoke test succeeds.

## Ablation study

The reproducible model-comparison package is in `experiments/ablation_study/`.
It currently compares four complete IU R2Gen test-set runs (590 matched
studies): the official CvT2DistilGPT2 baseline, BioViL-T with a frozen
DistilGPT2 decoder and linear projector (B1), BioViL-T with a fine-tuned
DistilGPT2 decoder (B2), and the BioViL-T clinical projector/auxiliary-loss
variant. The report-generation decoder is the IU-trained CvT2DistilGPT2
implementation; BioViL-T is the independent Microsoft chest-X-ray image
encoder. The ablation table preserves BLEU, ROUGE, METEOR, CIDEr, CheXbert and
availability status for unavailable clinical scorers.

Future feature extractors, projectors and decoders are registered in exactly
three files: `vision_encoders.py`, `projectors.py` and `llm_decoders.py`.
Qwen2 and Llama are declared as future levels but require a tested visual-token
adapter and local checkpoint before they can be evaluated fairly. Downloaded
weights, private datasets and generated predictions are intentionally excluded
from version control.

```powershell
python experiments\ablation_study\ablation.py --mode validate
python experiments\ablation_study\ablation.py --mode collect
python experiments\ablation_study\ablation.py --mode registry
```

