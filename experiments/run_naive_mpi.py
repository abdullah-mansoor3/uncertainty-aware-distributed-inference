"""MPI-native naive parallel runner.

Master (rank 0) orchestrates decomposition and schedules subtasks round-robin
to worker ranks (1..N-1). Workers run `src.worker.mpi_worker.worker_loop` and
respond with `result` messages.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.aggregator import aggregate_attributions, merge_outputs
from src.modules.decomposition import decompose_prompt
from src.modules.explanation import compute_local_attribution
from src.modules.inference import generate, load_model
from src.modules.uncertainty import compute_auroc, compute_erce, compute_pro_score
from src.scheduler.naive import NaiveParallelScheduler
from src.utils.config_loader import config_hash, load_config
from src.utils.metrics import (
    compute_bert_score,
    compute_bleu,
    compute_latency_stats,
    compute_meteor,
    compute_rouge,
    validate_required_metrics,
)
from src.utils.mpi_networking import worker_ranks, recv_message, send_task, probe_workers, wait_for_workers

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run naive MPI parallel pipeline")
    parser.add_argument("--config", required=True, help="Path to cluster config YAML")
    parser.add_argument("--dataset", required=True, help="Path to processed dataset JSONL")
    parser.add_argument("--output", required=True, help="Path to output results JSONL")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                samples.append(json.loads(text))
    return samples


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def build_contextualized_subtask_prompt(original_prompt: str, subtask_text: str) -> str:
    """Preserve global context while focusing generation on one subtask."""
    return (
        "You are solving one subtask from a larger user request.\n"
        "Use the full context below, but only answer the specific subtask.\n\n"
        f"FULL REQUEST:\n{original_prompt}\n\n"
        f"SUBTASK TO SOLVE:\n{subtask_text}\n\n"
        "Return only the subtask answer."
    )


def compute_decomposition_alignment_score(predicted_subtasks: List[str], reference_subtasks: List[str]) -> float:
    """Compute token-overlap F1 between predicted and reference subtasks."""
    if not reference_subtasks:
        return float("nan")
    pred_tokens = set(" ".join(predicted_subtasks).lower().split())
    ref_tokens = set(" ".join(reference_subtasks).lower().split())
    if not pred_tokens or not ref_tokens:
        return float("nan")
    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def main() -> None:
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    config = load_config(args.config)
    random_seed = int(config["runtime"]["random_seed"]) if config.get("runtime") else 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    model_cfg = config["model"]
    master_cfg = config["nodes"]["master"]
    scheduler_cfg = config["scheduler"]
    metric_ok, missing_metrics = validate_required_metrics(["rouge", "meteor", "bleu", "bert"])
    if not metric_ok:
        if rank == 0:
            LOGGER.error(
                "Missing critical metric dependencies: %s. Exiting without running experiments.",
                ", ".join(missing_metrics),
            )
        return

    if rank == 0:
        # Master
        llm = load_model(
            model_path=str(model_cfg["path"]),
            n_threads=int(master_cfg["n_threads"]),
            n_ctx=int(model_cfg["n_ctx"]),
            logprobs=int(model_cfg["logprobs"]),
        )

        samples = load_dataset(args.dataset)
        scheduler = NaiveParallelScheduler()
        output_path = Path(args.output)
        if output_path.exists():
            output_path.unlink()

        run_latencies: List[float] = []
        calibration_scores: List[float] = []
        calibration_correctness: List[float] = []

        expected_workers = worker_ranks()
        available_workers = wait_for_workers(expected_workers, timeout_s=120.0)
        if not available_workers:
            LOGGER.warning("No MPI workers found (size=%s). Falling back to serial execution.", size)
        else:
            LOGGER.info("MPI workers ready: %s", available_workers)

        for sample in samples:
            sample_start = time.perf_counter()
            prompt = str(sample.get("original_prompt", "")).strip()
            references = [str(item) for item in sample.get("ground_truth", [])]
            decomposition_references = [str(item) for item in sample.get("decomposition_ground_truth", []) if str(item).strip()]

            step_latencies_ms: Dict[str, int] = {
                "decomposition": 0,
                "probe": 0,
                "scheduling": 0,
                "dispatch_send": 0,
                "dispatch_wait": 0,
                "local_inference": 0,
                "attribution": 0,
                "aggregation": 0,
                "metrics": 0,
            }
            per_subtask_latencies_ms: List[Dict[str, Any]] = []

            decomp_start = time.perf_counter()
            subtasks = decompose_prompt(prompt=prompt, llm=llm)
            step_latencies_ms["decomposition"] = int((time.perf_counter() - decomp_start) * 1000)

            schedule_start = time.perf_counter()
            scheduled = scheduler.schedule(subtasks)
            step_latencies_ms["scheduling"] = int((time.perf_counter() - schedule_start) * 1000)

            subtask_count = max(1, len(scheduled))
            per_subtask_max_tokens = max(32, int(model_cfg["max_tokens"]) // subtask_count)
            probe_tokens = min(int(scheduler_cfg.get("probe_tokens", 5)), per_subtask_max_tokens)

            for item in scheduled:
                exec_prompt = build_contextualized_subtask_prompt(prompt, str(item["text"]))
                item["execution_prompt"] = exec_prompt
                probe_start = time.perf_counter()
                probe = generate(
                    llm=llm,
                    prompt=exec_prompt,
                    max_tokens=probe_tokens,
                    temperature=float(model_cfg.get("temperature", 0.0)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    logprobs=int(model_cfg.get("logprobs", 10)),
                    max_inference_ms=int(model_cfg.get("max_inference_ms", 0)),
                )
                step_latencies_ms["probe"] += int((time.perf_counter() - probe_start) * 1000)
                item["pro_score"] = compute_pro_score(probe.get("pro_logprobs") or probe.get("logprobs", []), adaptive_k=True)

            # Respect naive round-robin assignment: run node_a tasks locally and node_b on workers.
            local_tasks = [task for task in scheduled if str(task.get("assigned_node", "node_a")) == "node_a" or not available_workers]
            remote_tasks = [task for task in scheduled if str(task.get("assigned_node", "node_a")) == "node_b" and available_workers]

            results_by_id: Dict[int, Dict[str, Any]] = {}

            for subtask in local_tasks:
                local_start = time.perf_counter()
                generation = generate(
                    llm=llm,
                    prompt=str(subtask["execution_prompt"]),
                    max_tokens=per_subtask_max_tokens,
                    temperature=float(model_cfg.get("temperature", 0.0)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    logprobs=int(model_cfg.get("logprobs", 10)),
                    max_inference_ms=int(model_cfg.get("max_inference_ms", 0)),
                )
                step_latencies_ms["local_inference"] += int((time.perf_counter() - local_start) * 1000)
                output = str(generation.get("text", ""))
                attribution_start = time.perf_counter()
                try:
                    attribution = compute_local_attribution(subtask=str(subtask["text"]), output=output, llm=llm, nlp=None)
                except Exception:
                    attribution = []
                step_latencies_ms["attribution"] += int((time.perf_counter() - attribution_start) * 1000)

                subtask_id = int(subtask["id"])
                item_latency_ms = int(generation.get("latency_ms", 0))
                results_by_id[subtask_id] = {
                    "id": subtask_id,
                    "output": output,
                    "attribution": attribution,
                    "latency_ms": item_latency_ms,
                    "worker_rank": 0,
                    "effective_node": "node_a",
                    "round_trip_ms": item_latency_ms,
                }
                per_subtask_latencies_ms.append(
                    {
                        "id": subtask_id,
                        "assigned_node": "node_a",
                        "effective_node": "node_a",
                        "inference_ms": item_latency_ms,
                        "round_trip_ms": item_latency_ms,
                    }
                )

            pending = list(remote_tasks)
            in_flight = 0
            worker_cycle = iter(available_workers) if available_workers else iter([])
            send_started_at: Dict[int, float] = {}

            send_start = time.perf_counter()
            while pending and available_workers:
                try:
                    w = next(worker_cycle)
                except StopIteration:
                    worker_cycle = iter(available_workers)
                    w = next(worker_cycle)
                task = pending.pop(0)
                payload = {
                    "id": int(task["id"]),
                    "text": str(task["execution_prompt"]),
                    "max_tokens": per_subtask_max_tokens,
                    "temperature": float(model_cfg.get("temperature", 0.0)),
                }
                send_task(w, payload)
                send_started_at[int(task["id"])] = time.perf_counter()
                in_flight += 1
                if in_flight >= len(available_workers):
                    break
            step_latencies_ms["dispatch_send"] += int((time.perf_counter() - send_start) * 1000)

            wait_start = time.perf_counter()
            while in_flight > 0:
                msg = recv_message()
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "result":
                    payload = msg.get("payload", {})
                    subtask_id = int(payload.get("id", -1))
                    round_trip_ms = int((time.perf_counter() - send_started_at.get(subtask_id, time.perf_counter())) * 1000)
                    payload["round_trip_ms"] = round_trip_ms
                    payload["effective_node"] = "node_b"
                    results_by_id[subtask_id] = payload
                    per_subtask_latencies_ms.append(
                        {
                            "id": subtask_id,
                            "assigned_node": "node_b",
                            "effective_node": "node_b",
                            "inference_ms": int(payload.get("latency_ms", 0)),
                            "round_trip_ms": round_trip_ms,
                            "worker_rank": int(payload.get("worker_rank", -1)),
                        }
                    )
                    in_flight -= 1
                    if pending:
                        try:
                            w = next(worker_cycle)
                        except StopIteration:
                            worker_cycle = iter(available_workers)
                            w = next(worker_cycle)
                        task = pending.pop(0)
                        send_payload = {
                            "id": int(task["id"]),
                            "text": str(task["execution_prompt"]),
                            "max_tokens": per_subtask_max_tokens,
                            "temperature": float(model_cfg.get("temperature", 0.0)),
                        }
                        send_task(w, send_payload)
                        send_started_at[int(task["id"])] = time.perf_counter()
                        in_flight += 1
            step_latencies_ms["dispatch_wait"] = int((time.perf_counter() - wait_start) * 1000)

            ordered_ids = [int(item["id"]) for item in scheduled]
            ordered_results = [results_by_id[subtask_id] for subtask_id in ordered_ids]

            outputs = [str(item.get("output", "")) for item in ordered_results]
            uncertainty_scores = [float(item.get("pro_score", float("nan"))) for item in scheduled]
            valid_uncertainty_scores = [score for score in uncertainty_scores if not np.isnan(score)]
            local_maps = [item.get("attribution", []) for item in ordered_results]
            aggregation_start = time.perf_counter()
            global_map = aggregate_attributions(local_maps, uncertainty_scores)
            step_latencies_ms["aggregation"] = int((time.perf_counter() - aggregation_start) * 1000)

            merged_output = merge_outputs(outputs, [int(item["id"]) for item in scheduled])
            metrics_start = time.perf_counter()
            rouge = compute_rouge(merged_output, references)
            meteor = compute_meteor(merged_output, references)
            bleu = compute_bleu(merged_output, references)
            bert = compute_bert_score(merged_output, references)
            step_latencies_ms["metrics"] = int((time.perf_counter() - metrics_start) * 1000)
            decomposition_alignment_score = compute_decomposition_alignment_score(
                [str(item.get("text", "")) for item in scheduled],
                decomposition_references,
            )
            sample_correctness = float(rouge["rouge1"]) if not np.isnan(float(rouge["rouge1"])) else 0.0
            sample_uncertainty = float(np.mean(valid_uncertainty_scores)) if valid_uncertainty_scores else float("nan")
            if not np.isnan(sample_uncertainty):
                calibration_scores.append(sample_uncertainty)
                calibration_correctness.append(sample_correctness)
            running_erce = compute_erce(calibration_scores, calibration_correctness, n_bins=int(config["evaluation"]["n_bins"]))
            running_auroc = compute_auroc(calibration_scores, calibration_correctness)

            latency_ms = int((time.perf_counter() - sample_start) * 1000)
            run_latencies.append(latency_ms)

            record = {
                "id": sample.get("id"),
                "pipeline": "naive_mpi",
                "original_prompt": prompt,
                "subtasks": [str(item.get("text", "")) for item in scheduled],
                "routing": {f"subtask_{idx}": str(item.get("assigned_node", "node_a")) for idx, item in enumerate(scheduled)},
                "uncertainty_scores": uncertainty_scores,
                "decomposition_alignment_score": decomposition_alignment_score,
                "outputs": outputs,
                "merged_output": merged_output,
                "attribution_vectors": local_maps,
                "global_attribution": global_map,
                "correctness": {
                    "rouge1": rouge["rouge1"],
                    "rougeL": rouge["rougeL"],
                    "meteor": meteor,
                    "bleu": bleu,
                    "bert": bert,
                },
                "latency_ms": latency_ms,
                "step_latencies_ms": step_latencies_ms,
                "per_subtask_latencies_ms": sorted(per_subtask_latencies_ms, key=lambda item: int(item.get("id", -1))),
                "node_latencies_ms": {
                    "node_a": sum(int(item.get("latency_ms", 0)) for item in ordered_results if int(item.get("worker_rank", 0)) == 0),
                    "node_b": sum(int(item.get("latency_ms", 0)) for item in ordered_results if int(item.get("worker_rank", 0)) != 0),
                },
                "running_erce": running_erce,
                "running_auroc": running_auroc,
                "fallback": False,
                "config_hash": config_hash(config),
                "model_path": model_cfg["path"],
                "quantization": "Q4_K_M",
            }
            append_jsonl(output_path, record)
            LOGGER.info("Processed sample id=%s latency=%sms", sample.get("id"), latency_ms)

        latency_stats = compute_latency_stats(run_latencies)
        final_erce = compute_erce(calibration_scores, calibration_correctness, n_bins=int(config["evaluation"]["n_bins"]))
        final_auroc = compute_auroc(calibration_scores, calibration_correctness)
        LOGGER.info(
            "Naive MPI pipeline completed. Samples=%s mean_latency=%.2f ERCE=%.4f AUROC=%.4f",
            len(run_latencies),
            latency_stats["mean"],
            final_erce,
            final_auroc,
        )

    else:
        # Worker ranks run the worker loop defined in src/worker/mpi_worker.py
        from src.worker.mpi_worker import worker_loop

        worker_loop()


if __name__ == "__main__":
    main()
