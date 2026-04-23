# Uncertainty-Aware Adaptive Distributed Inference on Heterogeneous CPU Clusters

Inference-only ANN + PDC project for comparing three pipelines on heterogeneous CPU nodes:

- Serial baseline (single-node master execution)
- Naive parallel (round-robin assignment)
- Uncertainty-aware parallel (PRO-based routing + RTT fallback)

## Current implementation status

Implemented end-to-end:

- Prompt decomposition with LLM-first JSON extraction and rule-based fallback.
- PRO uncertainty scoring from token log-probability entropy.
- Serial, naive parallel, and uncertainty-aware experiment runners.
- Worker FastAPI server for remote subtask execution.
- Local token attribution plus master-side attribution aggregation.
- Per-sample incremental JSONL result logging.

Partially paper-parity (known approximations):

- ERCE and AUROC are implemented, but AUARC/AUPRC are not yet implemented.
- Dependency detection in decomposition is heuristic lexical overlap.
- SyntaxShap attribution is an approximation (not full Shapley), by design for CPU feasibility.

## Metric coverage by pipeline

All three runners log and/or summarize the same core evaluation dimensions:

- Correctness: ROUGE-1, ROUGE-L, METEOR, BLEU, BERTScore
- Latency: per-sample latency and node-level latency totals
- Uncertainty: PRO score per subtask
- Calibration diagnostics: ERCE, AUROC (run summary level)
- Attribution: local attribution vectors (all pipelines) and aggregated global attribution

Result files:

- results/results_serial.jsonl
- results/results_naive.jsonl
- results/results_adaptive.jsonl

## Repository map

- src/modules/decomposition.py
- src/modules/uncertainty.py
- src/modules/inference.py
- src/modules/explanation.py
- src/modules/aggregator.py
- src/scheduler/serial.py
- src/scheduler/naive.py
- src/scheduler/uncertainty_aware.py
- src/worker/worker_server.py
- experiments/run_serial.py
- experiments/run_naive.py
- experiments/run_adaptive.py

## Setup

1) Create environment and install requirements.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Install runtime extras used by worker/inference flow if missing.

```bash
pip install uvicorn
python -m spacy download en_core_web_sm
```

3) Download model.

```bash
export HF_TOKEN=hf_xxx
bash models/download_models.sh
```

4) Prepare datasets (canonical JSONL).

```bash
python experiments/prepare_datasets.py --sample-size 100 --seed 42
```

## Configuration before distributed runs

Edit configs/cluster_config.yaml with your actual two-node values:

- Replace localhost IPs with Tailscale IPs for master/worker.
- Keep worker port aligned with worker server launch port.
- Ensure model.path points to the downloaded GGUF file.
- Keep scheduler thresholds aligned with your experiment protocol.

## Running experiments

Serial baseline:

```bash
python experiments/run_serial.py \
  --config configs/cluster_config.yaml \
  --dataset data/processed/nq_open_100.jsonl \
  --output results/results_serial.jsonl
```

Start worker (on node_b):

```bash
python -m uvicorn src.worker.worker_server:app --host 0.0.0.0 --port 8001
```

Naive parallel (on node_a):

```bash
python experiments/run_naive.py \
  --config configs/cluster_config.yaml \
  --dataset data/processed/nq_open_100.jsonl \
  --output results/results_naive.jsonl
```

Uncertainty-aware parallel (on node_a):

```bash
python experiments/run_adaptive.py \
  --config configs/cluster_config.yaml \
  --dataset data/processed/nq_open_100.jsonl \
  --output results/results_adaptive.jsonl
```

Repeat each pipeline for:

- data/processed/nq_open_100.jsonl
- data/processed/mmlu_pro_100.jsonl
- data/processed/synthetic_prompts.jsonl

## Analysis workflow

Use notebooks for visual analysis:

- notebooks/analysis.ipynb
- notebooks/pro_rank_calibration.ipynb

Compare across pipelines:

- Mean/P95/P99 latency
- Correctness metrics (ROUGE/METEOR/BLEU/BERTScore)
- PRO-calibration behavior (ERCE, AUROC)
- Attribution consistency/coherence trends

## Relation to base papers and reports

This implementation follows the project specification derived from:

- copilot_context/base_papers/PARALLELPROMPT.pdf
- copilot_context/base_papers/Probabilities Are All You Need.pdf
- copilot_context/base_papers/Uncertainty in Language Models: Assessment through Rank-Calibration.pdf
- copilot_context/base_papers/SYNTAXSHAP.pdf

and your report set under:

- copilot_context/our_reports/

The core project purpose is implemented: run as-is serial inference, then decomposition-driven distributed inference, then compare quality/latency/calibration/attribution outcomes across the three pipelines.
