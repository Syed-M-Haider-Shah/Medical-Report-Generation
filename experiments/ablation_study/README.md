# Ablation study

This folder contains the controlled, reproducible comparison of chest X-ray feature extractors, visual-token projectors, and report-generation decoders. It is deliberately independent from the original training folders, so new ablations do not modify or overwrite prior checkpoints.

## Protocol

Every completed system must use:

- the same IU R2Gen benchmark test annotation;
- the same 590 study IDs and study-level image grouping;
- the same Findings target and preprocessing convention;
- the same metric implementation and version record;
- an explicit local checkpoint path, training regime, and prediction hash.

`ablation.py` validates the shared test order and writes one CSV row per completed experiment. Missing clinical metrics remain empty and are accompanied by an exact `metric_availability` explanation.

## Component registries

The study is organized into three files. Add new levels to the relevant file rather than creating a new script for each model family.

| Registry | Responsibility |
|---|---|
| `vision_encoders.py` | Feature dimensions, token layouts, input size, checkpoint provenance, frozen/default behavior |
| `projectors.py` | Visual-token adapter architecture and parameter counts |
| `llm_decoders.py` | Decoder hidden size, visual interface, checkpoint policy, compatibility boundary |

The current registry includes CvT-21, BioViL-T, linear and MLP projectors, and the IU DistilGPT2 decoder. Qwen2 and Llama are declared as planned levels but remain blocked until their visual-token adapter and local checkpoint are tested. This prevents a text-only decoder from being inserted into an incompatible cross-attention interface.

## Completed runs

| Experiment | Encoder | Projector | Decoder/training |
|---|---|---|---|
| `cvt2distilgpt2_official` | CvT-21 | Original linear 384 -> 768 | Original IU DistilGPT2 checkpoint |
| `biovilt_distilgpt2_b1` | Frozen BioViL-T | Linear 512 -> 768 | Projector-only |
| `biovilt_distilgpt2_b2` | Frozen BioViL-T | Linear 512 -> 768 | Projector plus decoder fine-tuning |
| `biovilt_distilgpt2_clinical` | Frozen BioViL-T | MLP 512 -> 1024 -> 768 plus clinical head | Report CE plus CheXbert auxiliary loss |

The four runs are descriptive and exploratory. The clinical variant produced a collapsed report pattern in the recorded run and should be treated as a negative ablation until retrained with a corrected generation setup.

## Commands

From the project root:

```powershell
python experiments\ablation_study\ablation.py --mode registry
python experiments\ablation_study\ablation.py --mode validate
python experiments\ablation_study\ablation.py --mode collect
python experiments\ablation_study\ablation.py --mode show
```

The output table is `outputs/ablation_results.csv`. The CSV contains BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, CIDEr, BERTScore, CheXbert, RadGraph, GREEN, provenance, and metric availability fields.

## Adding a new experiment

1. Add or validate the component in the appropriate registry.
2. Use a new checkpoint directory; never overwrite an existing model.
3. Run the complete test split and save a standard metrics JSON plus prediction CSV.
4. Add one metadata entry to `EXPERIMENTS` in `ablation.py`.
5. Run `--mode validate`, then `--mode collect`.
6. Preserve the resulting CSV row and prediction hash in the experiment record.

Before publication, repeat each selected configuration with fixed seeds, report confidence intervals or paired bootstrap tests, inspect report diversity and pathology-level behavior, and evaluate unavailable clinical scorers in a reproducible environment.

## Artifact policy

Do not commit model weights, private datasets, or generated study-level predictions unless their licenses and data-use agreements permit redistribution. The registry stores paths and provenance; it does not perform implicit downloads.

