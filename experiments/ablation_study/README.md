# Reproducible chest-X-ray feature/decoder ablations

This is a separate paper experiment folder. The original CvT2DistilGPT2,
BioViL-T B1/B2 and clinical experiments remain untouched. The study has three
component registries, one file per factor:

* `vision_encoders.py`  encoder names, feature dimensions, token layouts and
  checkpoint provenance.
* `projectors.py`  linear, MLP and identity visual-token adapters.
* `llm_decoders.py`  decoder names, hidden sizes and visual-interface rules.

`ablation.py` remains the single validation/collection file. Add a new tested
component to the appropriate registry and add its completed run to its
`EXPERIMENTS` dictionary; do not create a second collector with different
metric or split conventions.

The current table contains four complete IU R2Gen test-set runs:

1. Original CvT-21 + original DistilGPT2 checkpoint.
2. Frozen BioViL-T + linear projector + frozen decoder (B1).
3. Frozen BioViL-T + linear projector + fine-tuned DistilGPT2 (B2).
4. Frozen BioViL-T + clinical MLP/projector + B2 decoder + CheXbert auxiliary loss.

Each row is required to contain exactly 590 unique studies in the same order.
Missing metrics remain blank and their exact availability/error status is stored
in `metric_availability`; unavailable metrics are never replaced by zero.

## Build the paper table

From the project root:

```powershell
python experiments\ablation_study\ablation.py --mode validate
python experiments\ablation_study\ablation.py --mode collect
python experiments\ablation_study\ablation.py --mode show
python experiments\ablation_study\ablation.py --mode registry
```

The consolidated file is:

```text
experiments\ablation_study\outputs\ablation_results.csv
```

The table includes BLEU-1/2/3/4, COCO-compatible ROUGE-L, ROUGE-1/2,
METEOR, CIDEr, BERTScore, CheXbert 5/14 micro and macro F1, RadGraph and GREEN,
plus model/projection/training provenance. All rows use the same report target,
test IDs and metric worker. `metric_availability` records that BERTScore,
RadGraph or GREEN were unavailable where their local checkpoint/integration was
not complete.

## Planned ablations

The controlled factors are:

| Factor | Current levels |
|---|---|
| Feature extractor | CvT-21; BioViL-T |
| Decoder | original frozen DistilGPT2; fine-tuned DistilGPT2 |
| Projector | original linear; new linear; clinical MLP |
| Auxiliary objective | report CE; report CE + CheXbert auxiliary loss |

The completed rows are exploratory, not a claim of superiority. The clinical
row collapsed to one report template and should be reported as a negative
ablation. Before publication, add matched random seeds, confidence intervals or
paired bootstrap tests, verify unavailable clinical scorers, inspect report
diversity and pathology-level behavior, and keep the IU test set untouched during
model selection.

To add a future run, first save its standard `metrics.json` and test prediction
CSV, then add one entry to `EXPERIMENTS` in `ablation.py` and rerun `--mode
validate`. This creates one new row without changing prior rows.

Qwen and Llama are listed as planned decoder levels, but their builders refuse
to run until a visual-token adapter and a local checkpoint are explicitly
implemented and tested. A decoder-only text model cannot be substituted for
the current DistilGPT2 cross-attention interface by changing a model name.
Every experiment must record the local checkpoint path, model revision when
available, seed, dataset split and preprocessing configuration. No registry
performs an implicit download or overwrites an existing checkpoint.

## GitHub upload

This workspace currently has no Git metadata or configured remote. The folder is
ready to commit, but upload requires a GitHub repository URL and authenticated
Git client. Do not commit downloaded model weights, private datasets or generated
test predictions unless their licenses and repository policy permit it.

