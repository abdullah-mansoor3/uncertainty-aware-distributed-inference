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
from src.modules.uncertainty import compute_auroc, compute_erce, compute_pro_score
from src.scheduler.uncertainty_aware import UncertaintyAwareScheduler
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


def build_contextualized_subtask_prompt(original_prompt: str, subtask_text: str) -> str:
    """Preserve full context while executing one decomposed subtask."""
    return (
        "You are solving one subtask from a larger request.\n"
        "Use the full request context below, but answer only the subtask.\n\n"
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


def count_tokens(llm: Any, text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    if hasattr(llm, "tokenize"):
        try:
            tokens = llm.tokenize(text.encode("utf-8"), add_bos=False)
            return int(len(tokens))
        except Exception:
            pass
    return len(text.split())


def get_local_world_ranks(comm: MPI.Comm) -> List[int]:
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    world_group = comm.Get_group()
    local_group = local_comm.Get_group()
    local_world_ranks = local_group.Translate_ranks(list(range(local_group.Get_size())), world_group)
    return [int(r) for r in local_world_ranks]


def schedule_subtasks(
    subtasks: List[Dict[str, Any]],
    subtask_tokens: List[int],
    local_ms_per_token: float,
    local_world_ranks: List[int],
    worker_ms_per_token: Dict[int, EMA],
    worker_rtt_ema: Dict[int, EMA],
    default_ms_per_token: float,
    default_rtt_ms: float,
    workers: List[int],
    uncertainty_threshold: float,
    tie_margin: float,
) -> List[Dict[str, Any]]:
    worker_ends: Dict[int, float] = {w: 0.0 for w in workers}
    local_end = 0.0
    scheduled: List[Dict[str, Any]] = []

    for idx, subtask in enumerate(subtasks):
        tokens = subtask_tokens[idx] if idx < len(subtask_tokens) else count_tokens(None, str(subtask.get("text", "")))
        pro_score = float(subtask.get("pro_score", 1.0))

        local_duration = max(tokens, 1) * local_ms_per_token
        local_finish = local_end + local_duration

        best_worker = None
        best_finish = local_finish

        for w in workers:
            per_token = worker_ms_per_token[w].value if worker_ms_per_token[w].value is not None else default_ms_per_token
            rtt = worker_rtt_ema[w].value if worker_rtt_ema[w].value is not None else default_rtt_ms
            if w in local_world_ranks:
                rtt = 0.0
            finish = worker_ends[w] + rtt + (max(tokens, 1) * per_token)
            if finish < best_finish:
                best_finish = finish
                best_worker = w

        # PRO-primary routing:
        # - High confidence of high uncertainty -> force node_a.
        # - High confidence of low uncertainty -> force worker path.
        # - Near threshold (tie zone) -> use cost-based tie-break.
        if pro_score > (uncertainty_threshold + tie_margin):
            local_end = local_finish
            scheduled.append(dict(subtask, assigned_rank=0, assigned_node="node_a"))
            continue

        if pro_score < (uncertainty_threshold - tie_margin):
            if best_worker is None:
                local_end = local_finish
                scheduled.append(dict(subtask, assigned_rank=0, assigned_node="node_a"))
            else:
                worker_ends[best_worker] = best_finish
                assigned_node = "node_a" if best_worker in local_world_ranks else "node_b"
                scheduled.append(dict(subtask, assigned_rank=int(best_worker), assigned_node=assigned_node))
            continue

        # Tie zone: use dynamic cost model.
        if best_worker is None:
            local_end = local_finish
            scheduled.append(dict(subtask, assigned_rank=0, assigned_node="node_a"))
        else:
            worker_ends[best_worker] = best_finish
            assigned_node = "node_a" if best_worker in local_world_ranks else "node_b"
            scheduled.append(dict(subtask, assigned_rank=int(best_worker), assigned_node=assigned_node))

    return scheduled


def should_decompose(decomposition_time_ms: float, estimated_workers_time_ms: float) -> bool:
    """Heuristic gate: decompose only if expected benefit outweighs cost.

    This simple heuristic uses decomposition time versus estimated worker
    processing time to determine if decomposition is worthwhile.
    """
    if estimated_workers_time_ms <= 0:
        return False
    # require at least a modest improvement to decompose
    return decomposition_time_ms < (0.5 * estimated_workers_time_ms)


def main() -> None:
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_world_ranks = get_local_world_ranks(comm)

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
        worker_rtt_ema: Dict[int, EMA] = {w: EMA(alpha=0.2) for w in available_workers}
        local_ema = EMA(alpha=0.2)
        calibration_scores: List[float] = []
        calibration_correctness: List[float] = []
        run_latencies: List[float] = []

        if available_workers:
            initial_rtts = probe_workers(timeout_s=5.0)
            for w, rtt in initial_rtts.items():
                if w in worker_rtt_ema:
                    worker_rtt_ema[w].update(rtt)

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

            # quick decomposition timing
            t0 = time.perf_counter()
            subtasks = decompose_prompt(prompt=prompt, llm=llm)
            decomposition_time_ms = (time.perf_counter() - t0) * 1000.0
            step_latencies_ms["decomposition"] = int(decomposition_time_ms)

            n_subtasks = len(subtasks)
            per_subtask_max_tokens = max(32, int(model_cfg["max_tokens"]) // max(1, n_subtasks))
            subtask_tokens = [count_tokens(llm, str(item.get("text", ""))) for item in subtasks]
            default_ms_per_token = float(model_cfg.get("max_inference_ms", 1000)) / max(1, int(model_cfg.get("max_tokens", 1)))
            avg_worker_ms_per_token = float(
                np.nanmean([e.value for e in worker_ema.values() if e.value is not None])
                or default_ms_per_token
            )
            estimated_workers_time_ms = sum(max(tok, 1) * avg_worker_ms_per_token for tok in subtask_tokens)
            decomposed=True
            if not should_decompose(decomposition_time_ms, estimated_workers_time_ms):
                decomposed = False
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
                step_latencies_ms["local_inference"] += int(generation.get("latency_ms", 0))
                metrics_start = time.perf_counter()
                rouge = compute_rouge(merged_output, references)
                meteor = compute_meteor(merged_output, references)
                bleu = compute_bleu(merged_output, references)
                bert = compute_bert_score(merged_output, references)
                step_latencies_ms["metrics"] = int((time.perf_counter() - metrics_start) * 1000)
                decomposition_alignment_score = compute_decomposition_alignment_score([prompt], decomposition_references)
                running_erce = compute_erce(calibration_scores, calibration_correctness, n_bins=int(config["evaluation"]["n_bins"]))
                running_auroc = compute_auroc(calibration_scores, calibration_correctness)

                latency_ms = int((time.perf_counter() - sample_start) * 1000)
                run_latencies.append(latency_ms)
                prompt_tokens = count_tokens(llm, prompt)
                local_ema.update(float(latency_ms) / max(prompt_tokens, 1))
                record = {
                    "id": sample.get("id"),
                    "pipeline": "adaptive_mpi",
                    "original_prompt": prompt,
                    "subtasks": [prompt],
                    "routing": {"subtask_0": "node_a"},
                    "uncertainty_scores": [],
                    "decomposition_alignment_score": decomposition_alignment_score,
                    "outputs": [merged_output],
                    "merged_output": merged_output,
                    "attribution_vectors": [],
                    "global_attribution": [],
                    "correctness": {"rouge1": rouge["rouge1"], "rougeL": rouge["rougeL"], "meteor": meteor, "bleu": bleu, "bert": bert},
                    "latency_ms": latency_ms,
                    "step_latencies_ms": step_latencies_ms,
                    "per_subtask_latencies_ms": [],
                    "node_latencies_ms": {"node_a": latency_ms, "node_b": 0},
                    "running_erce": running_erce,
                    "running_auroc": running_auroc,
                    "fallback_to_serial": True,
                    "fallback": True,
                    "locally_processed": True,
                    "config_hash": config_hash(config),
                    "model_path": model_cfg["path"],
                    "quantization": "Q4_K_M",
                    "decomposed": decomposed,
                }
                append_jsonl(output_path, record)
                continue

            # if we reached here, dispatch subtasks to workers similarly to naive
            scheduler = UncertaintyAwareScheduler(uncertainty_threshold=float(scheduler_cfg.get("uncertainty_threshold", 0.5)), network_feasible_fn=lambda: True)
            scored_subtasks = []
            probe_tokens_budget = min(int(scheduler_cfg.get("probe_tokens", 5)), per_subtask_max_tokens)
            for subtask in subtasks:
                subtask_text = str(subtask["text"])
                exec_prompt = build_contextualized_subtask_prompt(prompt, subtask_text)
                # probe using generate with small tokens to compute pro_score
                probe_start = time.perf_counter()
                probe = generate(
                    llm=llm,
                    prompt=exec_prompt,
                    max_tokens=probe_tokens_budget,
                    temperature=float(model_cfg.get("temperature", 0.0)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    logprobs=int(model_cfg.get("logprobs", 10)),
                    max_inference_ms=int(model_cfg.get("max_inference_ms", 0)),
                )
                step_latencies_ms["probe"] += int((time.perf_counter() - probe_start) * 1000)
                probe_tokens = count_tokens(llm, str(subtask.get("text", "")))
                local_ema.update(float(probe.get("latency_ms", 0)) / max(probe_tokens, 1))
                pro_score = compute_pro_score(probe.get("pro_logprobs") or probe.get("logprobs", []), adaptive_k=True)
                item = dict(subtask)
                item["pro_score"] = pro_score
                item["execution_prompt"] = exec_prompt
                scored_subtasks.append(item)

            schedule_start = time.perf_counter()
            scheduled = scheduler.schedule(scored_subtasks)
            step_latencies_ms["scheduling"] = int((time.perf_counter() - schedule_start) * 1000)
            routing_proposed_by_pro = {f"subtask_{idx}": str(item.get("assigned_node", "node_a")) for idx, item in enumerate(scheduled)}

            default_rtt_ms = float(np.nanmean([e.value for e in worker_rtt_ema.values() if e.value is not None]) or 50.0)
            local_ms_per_token = local_ema.value if local_ema.value is not None else default_ms_per_token
            scheduled = schedule_subtasks(
                subtasks=scheduled,
                subtask_tokens=subtask_tokens,
                local_ms_per_token=local_ms_per_token,
                local_world_ranks=local_world_ranks,
                worker_ms_per_token=worker_ema,
                worker_rtt_ema=worker_rtt_ema,
                default_ms_per_token=default_ms_per_token,
                default_rtt_ms=default_rtt_ms,
                workers=available_workers,
                uncertainty_threshold=float(scheduler_cfg.get("uncertainty_threshold", 0.5)),
                tie_margin=float(scheduler_cfg.get("uncertainty_tie_margin", 0.05)),
            )

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
                step_latencies_ms["local_inference"] += int(generation.get("latency_ms", 0))
                metrics_start = time.perf_counter()
                rouge = compute_rouge(merged_output, references)
                meteor = compute_meteor(merged_output, references)
                bleu = compute_bleu(merged_output, references)
                bert = compute_bert_score(merged_output, references)
                step_latencies_ms["metrics"] = int((time.perf_counter() - metrics_start) * 1000)
                decomposition_alignment_score = compute_decomposition_alignment_score([prompt], decomposition_references)
                running_erce = compute_erce(calibration_scores, calibration_correctness, n_bins=int(config["evaluation"]["n_bins"]))
                running_auroc = compute_auroc(calibration_scores, calibration_correctness)
                latency_ms = int((time.perf_counter() - sample_start) * 1000)
                run_latencies.append(latency_ms)
                prompt_tokens = count_tokens(llm, prompt)
                local_ema.update(float(latency_ms) / max(prompt_tokens, 1))
                record = {
                    "id": sample.get("id"),
                    "pipeline": "adaptive_mpi",
                    "original_prompt": prompt,
                    "subtasks": [prompt],
                    "routing": {"subtask_0": "node_a"},
                    "uncertainty_scores": [],
                    "decomposition_alignment_score": decomposition_alignment_score,
                    "outputs": [merged_output],
                    "merged_output": merged_output,
                    "attribution_vectors": [],
                    "global_attribution": [],
                    "correctness": {"rouge1": rouge["rouge1"], "rougeL": rouge["rougeL"], "meteor": meteor, "bleu": bleu, "bert": bert},
                    "latency_ms": latency_ms,
                    "step_latencies_ms": step_latencies_ms,
                    "per_subtask_latencies_ms": [],
                    "node_latencies_ms": {"node_a": latency_ms, "node_b": 0},
                    "running_erce": running_erce,
                    "running_auroc": running_auroc,
                    "fallback_to_serial": True,
                    "fallback": True,
                    "locally_processed": True,
                    "config_hash": config_hash(config),
                    "model_path": model_cfg["path"],
                    "quantization": "Q4_K_M",
                    "decomposed": decomposed,
                }
                append_jsonl(output_path, record)
                continue

            local_task_ids = [int(item.get("id", -1)) for item in scheduled if int(item.get("assigned_rank", 0)) == 0]
            locally_processed = all(
                int(item.get("assigned_rank", 0)) in local_world_ranks for item in scheduled
            )

            if local_task_ids and len(local_task_ids) == len(scheduled):
                ordered_results = []
                for item in scheduled:
                    subtask_text = str(item.get("text", ""))
                    t0 = time.perf_counter()
                    generation = generate(
                        llm=llm,
                        prompt=str(item.get("execution_prompt", subtask_text)),
                        max_tokens=per_subtask_max_tokens,
                        temperature=float(model_cfg.get("temperature", 0.0)),
                        top_p=float(model_cfg.get("top_p", 1.0)),
                        logprobs=int(model_cfg.get("logprobs", 10)),
                        max_inference_ms=int(model_cfg.get("max_inference_ms", 0)),
                    )
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    step_latencies_ms["local_inference"] += latency_ms
                    output = str(generation.get("text", ""))
                    try:
                        attr_start = time.perf_counter()
                        attribution = compute_local_attribution(subtask=subtask_text, output=output, llm=llm, nlp=None)
                        step_latencies_ms["attribution"] += int((time.perf_counter() - attr_start) * 1000)
                    except Exception:
                        attribution = []
                    ordered_results.append(
                        {
                            "id": int(item.get("id", -1)),
                            "output": output,
                            "logprobs": [float(v) for v in generation.get("logprobs", [])],
                            "tokens": [str(t) for t in generation.get("tokens", [])],
                            "attribution": attribution,
                            "latency_ms": latency_ms,
                            "worker_rank": 0,
                        }
                    )
                    per_subtask_latencies_ms.append(
                        {
                            "id": int(item.get("id", -1)),
                            "assigned_node": "node_a",
                            "effective_node": "node_a",
                            "inference_ms": latency_ms,
                            "round_trip_ms": latency_ms,
                            "worker_rank": 0,
                        }
                    )

                    subtask_id = int(item.get("id", -1))
                    token_count = subtask_tokens[subtask_id] if 0 <= subtask_id < len(subtask_tokens) else count_tokens(llm, subtask_text)
                    per_token_ms = float(latency_ms) / max(token_count, 1)
                    local_ema.update(per_token_ms)

                outputs = [str(item.get("output", "")) for item in ordered_results]
                uncertainty_scores = [float(item.get("pro_score", float("nan"))) for item in scheduled]
                valid_uncertainty_scores = [score for score in uncertainty_scores if not np.isnan(score)]
                local_maps = [item.get("attribution", []) for item in ordered_results]
                aggregation_start = time.perf_counter()
                global_map = aggregate_attributions(local_maps, uncertainty_scores)
                step_latencies_ms["aggregation"] = int((time.perf_counter() - aggregation_start) * 1000)

                merged_output = merge_outputs(outputs, [int(item.get("id", -1)) for item in scheduled])
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
                    "pipeline": "adaptive_mpi",
                    "original_prompt": prompt,
                    "subtasks": [str(item.get("text", "")) for item in scheduled],
                    "routing": {f"subtask_{idx}": "node_a" for idx, _ in enumerate(scheduled)},
                    "routing_proposed_by_pro": routing_proposed_by_pro,
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
                    "per_subtask_latencies_ms": sorted(per_subtask_latencies_ms, key=lambda value: int(value.get("id", -1))),
                    "node_latencies_ms": {"node_a": sum(int(item.get("latency_ms", 0)) for item in ordered_results), "node_b": 0},
                    "running_erce": running_erce,
                    "running_auroc": running_auroc,
                    "fallback_to_serial": False,
                    "fallback": False,
                    "locally_processed": True,
                    "config_hash": config_hash(config),
                    "model_path": model_cfg["path"],
                    "quantization": "Q4_K_M",
                    "decomposed": decomposed,
                }
                append_jsonl(output_path, record)
                LOGGER.info("Processed sample id=%s latency=%sms (local subtasks)", sample.get("id"), latency_ms)
                continue

            # dispatch similarly to naive pipeline: simple pipelined send/recv
            pending_by_worker: Dict[int, List[Dict[str, Any]]] = {w: [] for w in available_workers}
            local_tasks = [item for item in scheduled if int(item.get("assigned_rank", 0)) == 0]
            for item in scheduled:
                w = int(item.get("assigned_rank", 0))
                if w in pending_by_worker:
                    pending_by_worker[w].append(item)

            in_flight = 0
            results_by_id: Dict[int, Dict[str, Any]] = {}
            send_started_at: Dict[int, float] = {}

            for item in local_tasks:
                subtask_text = str(item.get("text", ""))
                t0 = time.perf_counter()
                generation = generate(
                    llm=llm,
                    prompt=str(item.get("execution_prompt", subtask_text)),
                    max_tokens=per_subtask_max_tokens,
                    temperature=float(model_cfg.get("temperature", 0.0)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    logprobs=int(model_cfg.get("logprobs", 10)),
                    max_inference_ms=int(model_cfg.get("max_inference_ms", 0)),
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                step_latencies_ms["local_inference"] += latency_ms
                output = str(generation.get("text", ""))
                try:
                    attr_start = time.perf_counter()
                    attribution = compute_local_attribution(subtask=subtask_text, output=output, llm=llm, nlp=None)
                    step_latencies_ms["attribution"] += int((time.perf_counter() - attr_start) * 1000)
                except Exception:
                    attribution = []
                subtask_id = int(item.get("id", -1))
                results_by_id[subtask_id] = {
                    "id": subtask_id,
                    "output": output,
                    "logprobs": [float(v) for v in generation.get("logprobs", [])],
                    "tokens": [str(t) for t in generation.get("tokens", [])],
                    "attribution": attribution,
                    "latency_ms": latency_ms,
                    "worker_rank": 0,
                }
                per_subtask_latencies_ms.append(
                    {
                        "id": subtask_id,
                        "assigned_node": "node_a",
                        "effective_node": "node_a",
                        "inference_ms": latency_ms,
                        "round_trip_ms": latency_ms,
                        "worker_rank": 0,
                    }
                )

                token_count = subtask_tokens[subtask_id] if 0 <= subtask_id < len(subtask_tokens) else count_tokens(llm, subtask_text)
                local_ema.update(float(latency_ms) / max(token_count, 1))

            dispatch_send_start = time.perf_counter()
            for w, tasks in pending_by_worker.items():
                if not tasks:
                    continue
                task = tasks.pop(0)
                payload = {
                    "id": int(task["id"]),
                    "text": str(task.get("execution_prompt", task["text"])),
                    "max_tokens": per_subtask_max_tokens,
                    "temperature": float(model_cfg.get("temperature", 0.0)),
                }
                send_task(w, payload)
                send_started_at[int(task["id"])] = time.perf_counter()
                in_flight += 1
            step_latencies_ms["dispatch_send"] += int((time.perf_counter() - dispatch_send_start) * 1000)

            dispatch_wait_start = time.perf_counter()
            while in_flight > 0:
                msg = recv_message()
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "result":
                    payload = msg.get("payload", {})
                    subtask_id = int(payload.get("id", -1))
                    results_by_id[subtask_id] = payload
                    round_trip_ms = int((time.perf_counter() - send_started_at.get(subtask_id, time.perf_counter())) * 1000)
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
                    # update worker EMA from reported latency
                    wrank = int(payload.get("worker_rank", -1))
                    lat = float(payload.get("latency_ms", 0))
                    if wrank in worker_ema:
                        token_count = subtask_tokens[subtask_id] if 0 <= subtask_id < len(subtask_tokens) else 1
                        worker_ema[wrank].update(lat / max(token_count, 1))
                    if wrank in worker_rtt_ema and subtask_id in send_started_at:
                        total_ms = (time.perf_counter() - send_started_at[subtask_id]) * 1000.0
                        rtt_ms = max(0.0, total_ms - lat)
                        worker_rtt_ema[wrank].update(rtt_ms)
                    in_flight -= 1
                    if wrank in pending_by_worker and pending_by_worker[wrank]:
                        task = pending_by_worker[wrank].pop(0)
                        payload = {
                            "id": int(task["id"]),
                            "text": str(task.get("execution_prompt", task["text"])),
                            "max_tokens": per_subtask_max_tokens,
                            "temperature": float(model_cfg.get("temperature", 0.0)),
                        }
                        send_task(wrank, payload)
                        send_started_at[int(task["id"])] = time.perf_counter()
                        in_flight += 1
            step_latencies_ms["dispatch_wait"] = int((time.perf_counter() - dispatch_wait_start) * 1000)

            ordered_ids = [int(item["id"]) for item in scheduled]
            ordered_results = [results_by_id[subtask_id] for subtask_id in ordered_ids]

            outputs = [str(item.get("output", "")) for item in ordered_results]
            uncertainty_scores = [float(item.get("pro_score", float("nan"))) for item in scheduled]
            valid_uncertainty_scores = [score for score in uncertainty_scores if not np.isnan(score)]
            local_maps = [item.get("attribution", []) for item in ordered_results]
            aggregation_start = time.perf_counter()
            global_map = aggregate_attributions(local_maps, uncertainty_scores)
            step_latencies_ms["aggregation"] = int((time.perf_counter() - aggregation_start) * 1000)

            merged_output = merge_outputs(outputs, ordered_ids)
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

            node_a_latency = sum(
                int(item.get("latency_ms", 0))
                for item in ordered_results
                if int(item.get("worker_rank", 0)) in local_world_ranks
            )
            node_b_latency = sum(
                int(item.get("latency_ms", 0))
                for item in ordered_results
                if int(item.get("worker_rank", 0)) not in local_world_ranks
            )
            record = {
                "id": sample.get("id"),
                "pipeline": "adaptive_mpi",
                "original_prompt": prompt,
                "subtasks": [str(item.get("text", "")) for item in scheduled],
                "routing": {f"subtask_{idx}": str(item.get("assigned_node", "node_a")) for idx, item in enumerate(scheduled)},
                "routing_proposed_by_pro": routing_proposed_by_pro,
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
                "per_subtask_latencies_ms": sorted(per_subtask_latencies_ms, key=lambda value: int(value.get("id", -1))),
                "node_latencies_ms": {"node_a": node_a_latency, "node_b": node_b_latency},
                "running_erce": running_erce,
                "running_auroc": running_auroc,
                "fallback_to_serial": False,
                "fallback": any(bool(item.get("fallback", False)) for item in ordered_results),
                "locally_processed": locally_processed,
                "config_hash": config_hash(config),
                "model_path": model_cfg["path"],
                "quantization": "Q4_K_M",
                "decomposed": decomposed,
            }
            append_jsonl(output_path, record)
            LOGGER.info("Processed sample id=%s latency=%sms", sample.get("id"), latency_ms)

        latency_stats = compute_latency_stats(run_latencies)
        final_erce = compute_erce(calibration_scores, calibration_correctness, n_bins=int(config["evaluation"]["n_bins"]))
        final_auroc = compute_auroc(calibration_scores, calibration_correctness)
        LOGGER.info(
            "Adaptive MPI pipeline completed. Samples=%s mean_latency=%.2f ERCE=%.4f AUROC=%.4f",
            len(samples),
            latency_stats["mean"],
            final_erce,
            final_auroc,
        )

    else:
        # worker
        from src.worker.mpi_worker import worker_loop

        worker_loop()


if __name__ == "__main__":
    main()
