# Chest X-ray report-generation experiments

Reproducible evaluation and ablation studies for chest X-ray report generation. The repository keeps model components, checkpoints, dataset references, experiment metadata, predictions, and metrics separate so that new comparisons can be added without changing the evaluation protocol.

## Scope

The project evaluates study-level report generation on IU X-Ray/Open-I using the established R2Gen benchmark annotation split. The current completed comparison contains **590 identical test studies** for every system. No new test split is sampled locally, and no result is presented as a claim of superiority without additional statistical testing.

The repository supports two complementary tracks:

- **Reference evaluation:** the frozen `ZrH42/Janus-Pro-CXR-Zero` checkpoint and the existing IU/MIMIC evaluation scripts.
- **Ablation study:** controlled combinations of vision encoder, visual-token projector, and language decoder.

## Current ablation systems

| ID | Vision encoder | Projector | Decoder/training | Status |
|---|---|---|---|---|
| `cvt2distilgpt2_official` | Original CvT-21 (384 x 384) | Original linear 384 -> 768 | IU-trained DistilGPT2 checkpoint | Complete |
| `biovilt_distilgpt2_b1` | Microsoft BioViL-T, frozen | Linear 512 -> 768 | Frozen IU DistilGPT2 decoder; projector-only training | Complete |
| `biovilt_distilgpt2_b2` | Microsoft BioViL-T, frozen | Linear 512 -> 768 | Fine-tuned IU DistilGPT2 decoder | Complete |
| `biovilt_distilgpt2_clinical` | Microsoft BioViL-T, frozen | MLP 512 -> 1024 -> 768 with clinical head | Report CE plus CheXbert auxiliary loss | Complete |

The language decoder in these completed experiments is the IU-trained CvT2DistilGPT2/DistilGPT2 implementation. BioViL-T is the independent Microsoft chest X-ray image encoder. CheXbert is used only in the clinical auxiliary-loss experiment and for clinical label evaluation.

## Results currently available

All rows below use the same 590-study IU R2Gen test set. Values are rounded for readability; the machine-readable table preserves full precision.

| System | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | CIDEr | CheXbert micro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official CvT2DistilGPT2 | 0.4723 | 0.3024 | 0.2227 | 0.1740 | 0.3750 | 0.1993 | 0.6850 | 0.5435 |
| BioViL-T B1 | 0.4285 | 0.2545 | 0.1794 | 0.1358 | 0.3408 | 0.1747 | 0.5278 | 0.5435 |
| BioViL-T B2 | **0.4894** | **0.3090** | 0.2171 | 0.1564 | 0.3701 | **0.2041** | 0.4776 | 0.5330 |
| BioViL-T clinical | 0.4266 | 0.2515 | 0.1767 | 0.1335 | 0.3384 | 0.1731 | 0.5254 | 0.5435 |

These are descriptive results from the completed runs, not a statistical significance claim. BERTScore and RadGraph were not available in the recorded environment, and GREEN was not configured; the CSV records the exact reasons rather than replacing missing values with zero.

Full precision and provenance: [`experiments/ablation_study/outputs/ablation_results.csv`](experiments/ablation_study/outputs/ablation_results.csv).

## Extensible ablation design

Future experiments are registered in exactly three component files:

- [`vision_encoders.py`](experiments/ablation_study/vision_encoders.py): encoder name, feature dimension, token layout, input size, and checkpoint provenance.
- [`projectors.py`](experiments/ablation_study/projectors.py): linear, MLP, identity, and future visual-token adapters.
- [`llm_decoders.py`](experiments/ablation_study/llm_decoders.py): decoder interface, hidden size, local checkpoint, and compatibility rules.

Qwen2, Llama, and additional encoders are deliberately registry entries until a tested visual-token adapter and local checkpoint are available. Changing a model name alone is not a valid comparison. Each completed run must also be added to `EXPERIMENTS` in [`ablation.py`](experiments/ablation_study/ablation.py), which remains the single collector for split checks and metrics.

## Reproduce the ablation table

From the project root:

```powershell
python experiments\ablation_study\ablation.py --mode registry
python experiments\ablation_study\ablation.py --mode validate
python experiments\ablation_study\ablation.py --mode collect
python experiments\ablation_study\ablation.py --mode show
```

`validate` requires every completed prediction file to contain 590 unique studies in the same order. `collect` writes one row per full experiment. Checkpoints and model files are never downloaded implicitly or overwritten by the registries.

## Data, checkpoints, and privacy

Large model weights, private datasets, generated predictions, and local caches are excluded from version control. Their paths and hashes are recorded in the experiment table when available. Follow the license and access requirements of IU X-Ray/Open-I, MIMIC-CXR, BioViL-T, CvT2DistilGPT2, DistilGPT2, Janus-Pro-CXR-Zero, and CheXbert before redistributing any artifact.

## Repository status

This repository is intentionally maintained as an evolving research codebase. New model families can be added through the three registries while preserving the same split, preprocessing, metric definitions, provenance fields, and CSV schema.

## Broader CXR ablation study

The audit-first multi-model framework is in [`experiments/cxr_ablation`](experiments/cxr_ablation/README.md). It is separate from the completed four-run table and is designed for BioViL-T, CXR-CLIP, CheXagent XraySigLIP/XrayCLIP, GLoRIA, MGCA, REFERS, RadPhi, Libra, RaDialog, LLaVA-Rad, XrayGPT, MAIRA-2 and later additions. It verifies official sources, licences, checkpoint access, feature representations and encoder-decoder compatibility before training. It does not silently replace gated or technically incompatible models.

