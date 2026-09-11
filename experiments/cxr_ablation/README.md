# CXR encoder-decoder ablation study

This package implements the audit-first framework for comparing chest-X-ray-specialised vision encoders, radiology language decoders, and cross-modal connectors. It is isolated under `experiments/cxr_ablation/` so the existing Janus, CvT2DistilGPT2, BioViL-T, and agentic experiments remain unchanged.

The study begins on IU X-Ray/OpenI and uses the existing R2Gen annotation. The same configuration interfaces later accept an official MIMIC-CXR test manifest without changing the model architecture.

## Scientific scope

The primary comparison separates four effects:

1. CXR/report-pretrained visual representation.
2. Radiology-specialised language modelling.
3. Connector architecture and trainable parameter budget.
4. Complete pretrained report-generation systems evaluated as external baselines.

The existing CvT2DistilGPT2 run is retained as `B0_CvT2DistilGPT2`, a conventional control. It is not presented as a CXR-specialised decoder. Full models such as CheXagent, Libra, RaDialog, LLaVA-Rad, XrayGPT and MAIRA-2 remain complete-model baselines unless their authors provide a scientifically separable decoder interface.

## Audit-first execution order

The first phase does not start long training. It verifies sources, licensing, checkpoints, preprocessing and compatibility:

```powershell
# Use the dedicated environment; do not alter the existing environments.
conda create -n cxr-ablation python=3.10 -y
conda activate cxr-ablation
pip install -r experiments\cxr_ablation\requirements_cxr_ablation.txt

# Inspect existing IU studies and verify no split leakage.
python experiments\cxr_ablation\scripts\inspect_dataset.py

# Generate the model availability and environment reports.
python experiments\cxr_ablation\scripts\audit_registry.py

# Generate the encoder x decoder compatibility matrix.
python experiments\cxr_ablation\scripts\build_compatibility_matrix.py

# Record local checkpoint/feature-extraction readiness.
python experiments\cxr_ablation\scripts\test_all_components.py

# Validate one configured experiment without long training.
python experiments\cxr_ablation\scripts\run_experiment.py --config experiments\cxr_ablation\configs\base_iu.yaml --dry-run

# Only after the audit is reviewed, attempt official downloads.
python experiments\cxr_ablation\scripts\download_models.py --encoders
```

The generated reports are:

- `outputs/model_availability_report.csv` and `.md`
- `outputs/environment_audit.json`
- `outputs/component_compatibility_matrix.csv`
- `outputs/encoder_feature_audit.csv`
- `outputs/dataset_audit.json`
- `outputs/download_status.json` after download attempts

Unknown or gated models are recorded with an explicit reason. No unofficial mirror, generic replacement, fabricated URL, or silent model substitution is allowed.

## Registry

All requested models are described in [`configs/model_registry.yaml`](configs/model_registry.yaml). It currently contains 13 encoder candidates, 5 decoder candidates and 4 complete-model/benchmark candidates. `null` official URLs mean that a source still requires verification; they are not guessed.

Known official sources currently recorded include Microsoft BioViL-T, Soombit CXR-CLIP, Stanford AIMI XraySigLIP/XrayCLIP/CheXagent/RadPhi, X-iZhang Libra, ChantalMP RaDialog and MBZUAI XrayGPT. The registry also records BenchX families (GLoRIA, MGCA, REFERS, MRM, MedKLIP, M-FLAG, PTUnifier and ConVIRT) as manual-verification entries until an exact official checkpoint and license are confirmed.

MAIRA-2 is recorded as a complete external model and not a component decoder. Its model card requires acceptance of research-only terms and does not permit model redistribution; the download script will never bypass that gate.

## Connector and training policy

Stage 1 is:

```text
encoder frozen + decoder frozen + freshly initialised MLP2Connector trainable
```

The default connector is token-wise `Linear -> GELU -> Dropout -> Linear -> LayerNorm` and maps `[B,N,D_encoder]` to `[B,K,D_decoder]` without truncating dimensions. Multi-view tokens are concatenated with a view mask; frontal and lateral images are not averaged in the primary experiment.

The later stages are planned as configuration-controlled runners:

- Stage 1: encoder ablation.
- Stage 1B: LoRA adaptation of the top three encoders.
- Stage 2: technically valid decoder ablation.
- Stage 3: predefined top-2 x top-2 cross-matching.
- Stage 4: connector ablation.
- Stage 5: complete pretrained-model benchmarks.

A stage must not run until its compatibility and dry-run gates pass. The runner records a manifest containing configuration hash, annotation hash, package versions, GPU, seed and checkpoint identifiers.

## Fair evaluation

The test set is never used for model selection, early stopping, connector design or hyperparameter tuning. Every final prediction record includes study ID, ground truth, prediction, encoder, decoder, connector, checkpoint and number of views. Metrics that cannot be installed or executed reliably are written as `NOT AVAILABLE` with the exact error.

Planned outputs include result summaries, per-study results, efficiency summaries, paired bootstrap confidence intervals, full-model data-exposure audit, failure logs and publication tables/figures. No numerical result is fabricated before the corresponding checkpoint has been downloaded, sanity-tested and evaluated.

## Model and data policy

Weights are downloaded only under `cache/models/<model_name>/` and are excluded from Git. IU X-Ray/OpenI and MIMIC-CXR files remain subject to their own access and data-use terms. Checkpoint licences are recorded before use; gated access is never bypassed.

