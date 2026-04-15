# uncertainty-aware-distributed-inference

Uncertainty-Aware Distributed Inference on Heterogeneous CPU Clusters

Overview
This project implements a lightweight distributed inference system for transformer-based language models under constrained hardware settings. The system focuses on uncertainty-aware task routing, where input prompts are decomposed into independent subtasks and dynamically assigned to different compute nodes based on predicted uncertainty and node capability.

The project combines concepts from:

* Parallel and Distributed Computing (PDC)
* Artificial Neural Networks (ANN)
* Uncertainty estimation and calibration in language models

The goal is to demonstrate how adaptive scheduling using uncertainty signals can improve latency, efficiency, and reliability in local CPU-based environments.

Key Features

* Subtask extraction from input prompts (rule-based + LLM-based)
* Lightweight LLM inference using quantized models (TinyLlama / Phi-2)
* Uncertainty estimation using multi-sample generation (entropy-based)
* Adaptive scheduler based on uncertainty and hardware capability
* Distributed execution across heterogeneous nodes
* Aggregation of outputs and basic explanation signals
* Evaluation of latency, throughput, and output quality

System Architecture

1. Input prompt is received by the master node
2. Prompt is decomposed into independent subtasks
3. Each subtask is assigned an uncertainty score
4. Scheduler routes subtasks to available worker nodes
5. Worker nodes perform inference and return outputs
6. Master aggregates outputs and evaluates performance

Project Structure (planned)

* master/
  Handles task decomposition, scheduling, and aggregation
* worker/
  Runs model inference and returns results
* models/
  Contains quantized GGUF models (not included in repo)
* utils/
  Utility scripts (uncertainty, parsing, evaluation)
* datasets/
  Small subsets of evaluation datasets
* experiments/
  Scripts for running benchmarks and comparisons

Setup Instructions

1. Clone the repository
2. Install dependencies (Python 3.10+ recommended)
3. Install and build llama.cpp for CPU inference
4. Download quantized models (GGUF format)
5. Configure worker nodes with model paths
6. Run master and worker scripts

Models
Due to hardware constraints, this project uses small quantized models:

* TinyLlama 1.1B (for low-memory nodes)
* Phi-2 / Mistral 3B (for higher-capacity nodes)

Datasets
The project uses small subsets (50–100 samples) of:

* NQ-Open
* TriviaQA
* SQuAD
* MMLU (subset)
* Synthetic structured prompts for task decomposition

Evaluation Metrics

* Accuracy / correctness (ROUGE, BLEU, or exact match)
* Uncertainty correlation with correctness
* Latency and tail latency (P95/P99)
* Throughput (tasks per second)
* Scheduling efficiency

Experiment Settings
The system is evaluated under three configurations:

1. Single-node inference (baseline)
2. Naive parallel execution
3. Uncertainty-aware distributed scheduling (proposed method)

Limitations

* Uses simplified uncertainty metrics (entropy-based)
* No large-scale training or fine-tuning
* Limited dataset size due to hardware constraints
* Approximate explanation methods (no full SHAP implementation)

Future Work

* Improved uncertainty estimation (rank-calibration metrics)
* Advanced task decomposition using stronger models
* Better attribution methods (e.g., SHAP or Integrated Gradients)
* Scaling to more nodes and larger datasets

Contributors

* Abdullah Bin Mansoor
* Abdul Moiz Qazi

License
This project is for academic use.
