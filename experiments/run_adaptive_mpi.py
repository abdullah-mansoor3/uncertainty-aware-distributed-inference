"""MPI-native uncertainty-aware adaptive runner.

Master (rank 0) decides whether to decompose and routes subtasks to workers
based on an efficient heuristic. The implementation uses lightweight EMA
telemetry for per-worker inference latency and RTT estimates collected from
worker replies to inform a simple decomposition-worthiness gate.
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
from src.modules.uncertainty import compute_pro_score
from src.scheduler.uncertainty_aware import UncertaintyAwareScheduler
from src.utils.config_loader import config_hash, load_config
from src.utils.metrics import compute_bert_score, compute_bleu, compute_latency_stats, compute_meteor, compute_rouge
from src.utils.mpi_networking import worker_ranks, recv_message, send_task, probe_workers, wait_for_workers

LOGGER = logging.getLogger(__name__)


class EMA:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1 - self.alpha) * self.value
        return self.value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive MPI parallel pipeline")
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


def should_decompose(decomposition_time_ms: float, avg_worker_ms: float, n_subtasks: int) -> bool:
    """Heuristic gate: decompose only if expected benefit outweighs cost.

    This simple heuristic uses decomposition time versus estimated worker
    processing time to determine if decomposition is worthwhile.
    """
    # if decomposition is very slow compared to expected per-subtask worker time,
    # avoid decomposition.
    estimated_workers_time = avg_worker_ms * n_subtasks
    # require at least a modest improvement to decompose
    return decomposition_time_ms < (0.5 * estimated_workers_time)


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

    if rank == 0:
        llm = load_model(
            model_path=str(model_cfg["path"]),
            n_threads=int(master_cfg["n_threads"]),
            n_ctx=int(model_cfg["n_ctx"]),
            logprobs=int(model_cfg["logprobs"]),
        )

        samples = load_dataset(args.dataset)
        output_path = Path(args.output)
        if output_path.exists():
            output_path.unlink()

        expected_workers = worker_ranks()
        available_workers = wait_for_workers(expected_workers, timeout_s=120.0)
        if not available_workers:
            LOGGER.warning("No MPI workers found (size=%s). Falling back to serial execution.", size)
        else:
            LOGGER.info("MPI workers ready: %s", available_workers)

        # telemetry EMAs per worker
        worker_ema: Dict[int, EMA] = {w: EMA(alpha=0.2) for w in available_workers}

        for sample in samples:
            sample_start = time.perf_counter()
            prompt = str(sample.get("original_prompt", "")).strip()
            references = [str(item) for item in sample.get("ground_truth", [])]

            # quick decomposition timing
            t0 = time.perf_counter()
            subtasks = decompose_prompt(prompt=prompt, llm=llm)
            decomposition_time_ms = (time.perf_counter() - t0) * 1000.0

            n_subtasks = len(subtasks)
            avg_worker_ms = float(np.nanmean([e.value for e in worker_ema.values() if e.value is not None]) or model_cfg.get("max_inference_ms", 1000))

            if not should_decompose(decomposition_time_ms, avg_worker_ms, n_subtasks):
                # run locally serially
                LOGGER.info("Skipping decomposition for sample id=%s due to cost gate", sample.get("id"))
                generation = generate(
                    llm=llm,
                    prompt=prompt,
                    max_tokens=int(model_cfg["max_tokens"]),
                    temperature=float(model_cfg.get("temperature", 0.0)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    logprobs=int(model_cfg.get("logprobs", 10)),
                    max_inference_ms=int(model_cfg.get("max_inference_ms", 0)),
                )
                merged_output = generation.get("text", "")
                rouge = compute_rouge(merged_output, references)
                meteor = compute_meteor(merged_output, references)
                bleu = compute_bleu(merged_output, references)
                bert = compute_bert_score(merged_output, references)

                latency_ms = int((time.perf_counter() - sample_start) * 1000)
                record = {
                    "id": sample.get("id"),
                    "pipeline": "adaptive_mpi",
                    "original_prompt": prompt,
                    "subtasks": [prompt],
                    "routing": {"subtask_0": "node_a"},
                    "uncertainty_scores": [],
                    "outputs": [merged_output],
                    "merged_output": merged_output,
                    "attribution_vectors": [],
                    "global_attribution": [],
                    "correctness": {"rouge1": rouge["rouge1"], "rougeL": rouge["rougeL"], "meteor": meteor, "bleu": bleu, "bert": bert},
                    "latency_ms": latency_ms,
                    "node_latencies_ms": {"node_a": latency_ms, "node_b": 0},
                    "fallback_to_serial": True,
                    "fallback": True,
                    "config_hash": config_hash(config),
                    "model_path": model_cfg["path"],
                    "quantization": "Q4_K_M",
                }
                append_jsonl(output_path, record)
                continue

            # if we reached here, dispatch subtasks to workers similarly to naive
            scheduler = UncertaintyAwareScheduler(uncertainty_threshold=float(scheduler_cfg.get("uncertainty_threshold", 0.5)), network_feasible_fn=lambda: True)
            scored_subtasks = []
            for subtask in subtasks:
                # probe using generate with small tokens to compute pro_score
                probe = generate(
                    llm=llm,
                    prompt=str(subtask["text"]),
                    max_tokens=min(int(scheduler_cfg.get("probe_tokens", 5)), int(model_cfg["max_tokens"])),
                    temperature=float(model_cfg.get("temperature", 0.0)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    logprobs=int(model_cfg.get("logprobs", 10)),
                    max_inference_ms=int(model_cfg.get("max_inference_ms", 0)),
                )
                pro_score = compute_pro_score(probe.get("pro_logprobs") or probe.get("logprobs", []), adaptive_k=True)
                item = dict(subtask)
                item["pro_score"] = pro_score
                scored_subtasks.append(item)

            scheduled = scheduler.schedule(scored_subtasks)

            if not available_workers:
                LOGGER.warning("No workers available; running adaptive sample locally")
                # fallback to local
                generation = generate(
                    llm=llm,
                    prompt=prompt,
                    max_tokens=int(model_cfg["max_tokens"]),
                    temperature=float(model_cfg.get("temperature", 0.0)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    logprobs=int(model_cfg.get("logprobs", 10)),
                    max_inference_ms=int(model_cfg.get("max_inference_ms", 0)),
                )
                merged_output = generation.get("text", "")
                rouge = compute_rouge(merged_output, references)
                meteor = compute_meteor(merged_output, references)
                bleu = compute_bleu(merged_output, references)
                bert = compute_bert_score(merged_output, references)
                latency_ms = int((time.perf_counter() - sample_start) * 1000)
                record = {
                    "id": sample.get("id"),
                    "pipeline": "adaptive_mpi",
                    "original_prompt": prompt,
                    "subtasks": [prompt],
                    "routing": {"subtask_0": "node_a"},
                    "uncertainty_scores": [],
                    "outputs": [merged_output],
                    "merged_output": merged_output,
                    "attribution_vectors": [],
                    "global_attribution": [],
                    "correctness": {"rouge1": rouge["rouge1"], "rougeL": rouge["rougeL"], "meteor": meteor, "bleu": bleu, "bert": bert},
                    "latency_ms": latency_ms,
                    "node_latencies_ms": {"node_a": latency_ms, "node_b": 0},
                    "fallback_to_serial": True,
                    "fallback": True,
                    "config_hash": config_hash(config),
                    "model_path": model_cfg["path"],
                    "quantization": "Q4_K_M",
                }
                append_jsonl(output_path, record)
                continue

            # dispatch similarly to naive pipeline: simple pipelined send/recv
            pending = list(scheduled)
            in_flight = 0
            results_by_id: Dict[int, Dict[str, Any]] = {}
            worker_cycle = iter(available_workers)

            while pending and available_workers:
                try:
                    w = next(worker_cycle)
                except StopIteration:
                    worker_cycle = iter(available_workers)
                    w = next(worker_cycle)
                task = pending.pop(0)
                payload = {"id": int(task["id"]), "text": str(task["text"]), "max_tokens": int(model_cfg["max_tokens"]), "temperature": float(model_cfg.get("temperature", 0.0))}
                send_task(w, payload)
                in_flight += 1
                if in_flight >= len(available_workers):
                    break

            while in_flight > 0:
                msg = recv_message()
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "result":
                    payload = msg.get("payload", {})
                    results_by_id[int(payload.get("id", -1))] = payload
                    # update worker EMA from reported latency
                    wrank = int(payload.get("worker_rank", -1))
                    lat = float(payload.get("latency_ms", 0))
                    if wrank in worker_ema:
                        worker_ema[wrank].update(lat)
                    in_flight -= 1
                    if pending:
                        try:
                            w = next(worker_cycle)
                        except StopIteration:
                            worker_cycle = iter(available_workers)
                            w = next(worker_cycle)
                        task = pending.pop(0)
                        payload = {"id": int(task["id"]), "text": str(task["text"]), "max_tokens": int(model_cfg["max_tokens"]), "temperature": float(model_cfg.get("temperature", 0.0))}
                        send_task(w, payload)
                        in_flight += 1

            ordered_ids = [int(item["id"]) for item in scheduled]
            ordered_results = [results_by_id[subtask_id] for subtask_id in ordered_ids]

            outputs = [str(item.get("output", "")) for item in ordered_results]
            uncertainty_scores = [float(item.get("pro_score", float("nan"))) for item in scheduled]
            local_maps = [item.get("attribution", []) for item in ordered_results]
            global_map = aggregate_attributions(local_maps, uncertainty_scores)

            merged_output = merge_outputs(outputs, ordered_ids)
            rouge = compute_rouge(merged_output, references)
            meteor = compute_meteor(merged_output, references)
            bleu = compute_bleu(merged_output, references)
            bert = compute_bert_score(merged_output, references)

            latency_ms = int((time.perf_counter() - sample_start) * 1000)

            record = {
                "id": sample.get("id"),
                "pipeline": "adaptive_mpi",
                "original_prompt": prompt,
                "subtasks": [str(item.get("text", "")) for item in scheduled],
                "routing": {f"subtask_{idx}": str(item.get("assigned_node", "node_a")) for idx, item in enumerate(scheduled)},
                "uncertainty_scores": uncertainty_scores,
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
                "node_latencies_ms": {"node_a": sum(int(item.get("latency_ms", 0)) for item in ordered_results if item.get("worker_rank") is None or item.get("worker_rank") == 0), "node_b": sum(int(item.get("latency_ms", 0)) for item in ordered_results if item.get("worker_rank") is not None and item.get("worker_rank") != 0)},
                "fallback_to_serial": False,
                "fallback": any(bool(item.get("fallback", False)) for item in ordered_results),
                "config_hash": config_hash(config),
                "model_path": model_cfg["path"],
                "quantization": "Q4_K_M",
            }
            append_jsonl(output_path, record)
            LOGGER.info("Processed sample id=%s latency=%sms", sample.get("id"), latency_ms)

    else:
        # worker
        from src.worker.mpi_worker import worker_loop

        worker_loop()


if __name__ == "__main__":
    main()
