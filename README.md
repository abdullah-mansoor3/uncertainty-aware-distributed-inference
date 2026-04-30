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

```bash
python data/prepare_parallelprompt.py --use-datasets-lib --verbose
```

ParallelPrompt dataset preparation
-------------------------------

Basic usage:

```bash
python data/prepare_parallelprompt.py --verbose
```

Load from a local parquet cache:

```bash
python data/prepare_parallelprompt.py --from-local data/raw/parallelprompt_full.parquet --verbose
```

Custom sample size and output:

```bash
python data/prepare_parallelprompt.py --sample-size 50 --output data/processed/parallelprompt_50.jsonl
```

Flags:

- `--sample-size` (int, default: 100) Number of rows to sample after filtering.
- `--seed` (int, default: 42) Random seed for sampling.
- `--output` (str, default: data/processed/parallelprompt_100.jsonl) Output JSONL path.
- `--raw-output` (str, default: data/raw/parallelprompt_full.parquet) Local parquet cache path.
- `--from-local` (str, default: None) Read from a local parquet file and skip download.
- `--use-datasets-lib` (flag) Use HuggingFace datasets library instead of direct parquet read.
- `--verbose` (flag) Enable debug-level logging.

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

MPI-native runners
------------------

Two new MPI-native runners are provided which use `mpi4py` and are intended
to be launched via `mpirun` or `mpiexec`. These runners preserve the serial
baseline semantics while enabling round-robin and uncertainty-aware routing
with reduced communication overhead compared to the HTTP worker path.

Run the MPI naive pipeline (master + workers):

```bash
mpirun -np 4 --hostfile configs/hostfile \
  python experiments/run_naive_mpi.py \
  --config configs/cluster_config.yaml \
  --dataset data/processed/nq_open_100.jsonl \
  --output results/results_naive_mpi.jsonl
```

Run the MPI adaptive pipeline (master + workers):

```bash
mpirun -np 4 --hostfile configs/hostfile \
  python experiments/run_adaptive_mpi.py \
  --config configs/cluster_config.yaml \
  --dataset data/processed/nq_open_100.jsonl \
  --output results/results_adaptive_mpi.jsonl
```

Design choices for MPI runners
-----------------------------

- New runner files: `experiments/run_naive_mpi.py` and
  `experiments/run_adaptive_mpi.py` keep the original HTTP-based runners
  unchanged and provide a clear, low-risk MPI path.
- Master/worker topology: the master is always MPI rank 0 and workers are
  ranks 1..N-1. Ranks >=1 run `src/worker/mpi_worker.py` to execute subtasks.
- Message protocol: lightweight dict messages with types `task`, `result`,
  `ping` and `shutdown` are exchanged via `mpi4py` `comm.send`/`comm.recv`.
- Decomposition gate (adaptive only): a conservative heuristic prevents
  decomposition when `decomposition_time` is large relative to expected
  worker processing time; this avoids wasting time when decomposition is
  costlier than local inference.
- Telemetry: master maintains simple EMA estimators for per-worker latencies
  observed in returned results; EMAs inform future routing and gate decisions.
- Communication efficiency: the MPI runners use coarse-grained task messages
  and a pipelined send/receive loop to keep messaging overhead below compute
  time whenever possible.

Notes
-----
- The MPI runners require `mpi4py` in the environment (`pip install mpi4py`) and
  an MPI runtime such as OpenMPI (install and hostfile setup is not shown here).
- The `configs/cluster_config.yaml` file now contains an `mpi` block used by
  the MPI runners; leave other config values (model path, scheduler thresholds)
  unchanged to preserve experiment comparability.

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
