# Uncertainty-Aware Adaptive Distributed Inference on Heterogeneous CPU Clusters

This repository contains the ANN + PDC project implementation for inference-only LLM evaluation on heterogeneous CPU nodes.

## What this project does

- Decomposes prompts into subtasks.
- Estimates uncertainty using PRO-style probability entropy.
- Runs serial baseline inference (and supports extension to parallel schedulers).
- Evaluates output quality and calibration behavior.
- Logs reproducible per-sample JSONL records for analysis.

## Core components

- `src/modules/decomposition.py`: prompt decomposition + fallback splitter.
- `src/modules/inference.py`: GGUF model load/generate wrapper over `llama-cpp-python`.
- `src/modules/uncertainty.py`: PRO score + calibration utilities.
- `src/modules/aggregator.py`: output/attribution merge helpers.
- `src/scheduler/serial.py`: serial baseline scheduler.
- `src/utils/metrics.py`: ROUGE/BLEU/METEOR/BERT/latency utilities.
- `experiments/run_serial.py`: end-to-end serial experiment runner.

## Repository structure

```text
uncertainty-aware-distributed-inference/
├── configs/
│   └── cluster_config.yaml
├── data/
│   ├── raw/                     # ignored in git (downloaded artifacts)
│   ├── processed/               # ignored in git (generated JSONL)
│   └── download_datasets.py
├── experiments/
│   └── run_serial.py
├── models/
│   ├── download_models.sh
│   └── *.gguf                   # ignored in git
├── notebooks/
│   └── pro_rank_calibration.ipynb
├── results/                     # ignored in git
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## Environment setup

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download model weights:

```bash
export HF_TOKEN=hf_xxx
bash models/download_models.sh
```

4. Download and preprocess datasets:

```bash
python data/download_datasets.py --samples 100 --seed 42
```

## Run serial baseline

```bash
python experiments/run_serial.py \
  --config configs/cluster_config.yaml \
  --dataset data/processed/nq_open_100.jsonl \
  --output results/results_serial.jsonl
```

## Notebook workflow (ANN Assignment 3)

Use `notebooks/pro_rank_calibration.ipynb` for a standalone workflow that can:

- download full datasets,
- create configurable subset files,
- download GGUF model weights,
- run generation pipeline from scratch,
- compute PRO + baseline uncertainty columns,
- report ERCE/AUROC comparisons.

It is designed to run on Colab or local Jupyter.

## Configuration

All runtime parameters are centralized in `configs/cluster_config.yaml`:

- model path and decoding params,
- node specs and thread counts,
- scheduler thresholds,
- evaluation settings.

## Testing

```bash
pytest -q
```

## Git hygiene

Large and generated files are intentionally excluded in `.gitignore`, including:

- downloaded datasets,
- processed JSONL outputs,
- GGUF model weights,
- local caches,
- `copilot_context/` reference materials.

This keeps the repository lightweight and safe to push to GitHub.

## Paper fidelity and explicit deviations

This project is implemented to be as close as possible to the base-paper logic described in `copilot_context/instructions.txt`.

### 1) PRO (Nguyen et al., 2025)

Implemented (paper-aligned):
- Uses top-K log-probabilities and entropy normalization $H/\log K$.
- Uses adaptive-K filtering by removing very low probabilities.
- Uses probe inference before full generation in pipeline flow.
- Uses thresholded uncertainty classification (`tau=0.5` default/configurable).

Custom/approximate logic (explicit):
- PRO extraction currently uses first-token top-K alternatives from `llama-cpp-python` completion objects when available (fallback to token logprobs if top-K map is unavailable).
- This is a practical API-level approximation, not a full reimplementation of all internals from the original paper code.

### 2) Rank-Calibration (Huang et al., EMNLP 2024)

Implemented (paper-aligned intent):
- ERCE-style calibration error over ranked uncertainty.
- AUROC for uncertainty-as-error ranking quality.

Custom/approximate logic (explicit):
- ERCE/AUROC operate on available correctness proxy (`rouge1` default in this repo) rather than every metric setting from the original paper suite.
- Current repo computes ERCE/AUROC directly; AUARC/AUPRC are planned next for full parity with broader rank-calibration reporting.

### 3) ParallelPrompt-inspired decomposition (Kolawole et al., 2025)

Implemented (paper-aligned intent):
- JSON schema-guided LLM decomposition first.
- Fallback decomposition when JSON parsing/LLM output fails.
- Dependency check and serial-only merge behavior.

Custom/approximate logic (explicit):
- Dependency detection currently uses lexical-overlap heuristics rather than full entity/output-conditional dependency analysis from the paper ecosystem.

### 4) SyntaxShap / Token-level attribution

Implemented:
- Project structure and interfaces exist for local and aggregated attribution modules.

Custom/approximate logic (explicit):
- Full SyntaxShap coalition machinery is not yet fully integrated into end-to-end experiment runners.

### About base-paper repositories

- I attempted to auto-discover GitHub repository links from the local paper PDFs in this environment but could not reliably extract them from embedded PDF text streams.
- Alignment here therefore follows your project instruction spec and paper formulas/logic represented in your context files.

## TODO (implemented vs remaining)

- [x] Serial pipeline with decomposition, inference, uncertainty scoring, and JSONL logging.
- [x] PRO score via normalized top-K entropy with probe inference path.
- [x] Full dataset download + subset generation workflow in notebook.
- [x] Colab-ready standalone notebook pipeline (`notebooks/pro_rank_calibration.ipynb`).
- [x] Git hygiene for datasets/models/caches/context artifacts.
- [ ] Add AUARC and AUPRC metrics with report-ready tables across datasets.
- [ ] Implement and benchmark naive parallel scheduler end-to-end.
- [ ] Implement and benchmark uncertainty-aware distributed scheduler end-to-end.
- [ ] Integrate full SyntaxShap-style attribution into experiment runner outputs.
- [ ] Add network RTT fallback experiments and comparative latency/quality analysis.
