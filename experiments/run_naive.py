"""Run the naive parallel inference pipeline.

Implements the ParallelPrompt-style decomposition stage with round-robin routing,
without uncertainty-aware routing decisions. It serves as the ANN+PDC baseline
parallel policy for comparison against serial and adaptive pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import spacy

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
from src.utils.metrics import compute_bert_score, compute_bleu, compute_latency_stats, compute_meteor, compute_rouge
from src.utils.networking import send_subtask_sync

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for naive pipeline runner."""
    parser = argparse.ArgumentParser(description="Run naive parallel inference pipeline")
    parser.add_argument("--config", required=True, help="Path to cluster config YAML")
    parser.add_argument("--dataset", required=True, help="Path to processed dataset JSONL")
    parser.add_argument("--output", required=True, help="Path to output results JSONL")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure logging level and output format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load canonical JSONL dataset records from disk."""
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
    """Append one JSON record to output JSONL and flush immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_nlp_pipeline() -> Any:
    """Load spaCy parser used for local attribution; fallback to blank pipeline."""
    try:
        return spacy.load("en_core_web_sm")
    except Exception as error:
        LOGGER.warning("Failed to load en_core_web_sm (%s). Falling back to spacy.blank('en').", error)
        return spacy.blank("en")


def run_local_subtask(subtask_text: str, llm: Any, nlp: Any, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one subtask locally and compute local attribution map."""
    generation = generate(
        llm=llm,
        prompt=subtask_text,
        max_tokens=int(model_cfg["max_tokens"]),
        temperature=float(model_cfg["temperature"]),
        top_p=float(model_cfg["top_p"]),
        logprobs=int(model_cfg["logprobs"]),
        max_inference_ms=int(model_cfg["max_inference_ms"]),
    )
    output = str(generation.get("text", ""))
    try:
        attribution = compute_local_attribution(subtask=subtask_text, output=output, llm=llm, nlp=nlp)
    except Exception as error:
        LOGGER.warning("Local attribution failed: %s", error)
        attribution = []
    return {
        "output": output,
        "logprobs": [float(value) for value in generation.get("logprobs", [])],
        "tokens": [str(token) for token in generation.get("tokens", [])],
        "attribution": attribution,
        "latency_ms": int(generation.get("latency_ms", 0)),
    }


def execute_subtask(
    subtask: Dict[str, Any],
    llm: Any,
    nlp: Any,
    model_cfg: Dict[str, Any],
    worker_cfg: Dict[str, Any],
    request_timeout_s: float,
) -> Dict[str, Any]:
    """Execute one scheduled subtask on assigned node with fallback to local."""
    subtask_id = int(subtask["id"])
    subtask_text = str(subtask["text"])
    assigned_node = str(subtask.get("assigned_node", "node_a"))

    if assigned_node == "node_b":
        payload = {
            "subtask_text": subtask_text,
            "max_tokens": int(model_cfg["max_tokens"]),
            "temperature": float(model_cfg["temperature"]),
        }
        try:
            response = send_subtask_sync(
                ip=str(worker_cfg["ip"]),
                port=int(worker_cfg["port"]),
                subtask=payload,
                timeout_s=request_timeout_s,
            )
            return {
                "id": subtask_id,
                "assigned_node": assigned_node,
                "effective_node": "node_b",
                "fallback": False,
                "output": str(response.get("output", "")),
                "logprobs": [float(value) for value in response.get("logprobs", [])],
                "tokens": [str(token) for token in response.get("tokens", [])],
                "attribution": response.get("attribution", []),
                "latency_ms": int(response.get("latency_ms", 0)),
            }
        except Exception as error:
            LOGGER.warning("Worker dispatch failed for subtask %s: %s. Falling back to node_a.", subtask_id, error)

    local = run_local_subtask(subtask_text=subtask_text, llm=llm, nlp=nlp, model_cfg=model_cfg)
    return {
        "id": subtask_id,
        "assigned_node": assigned_node,
        "effective_node": "node_a",
        "fallback": assigned_node == "node_b",
        **local,
    }


def main() -> None:
    """Execute naive round-robin pipeline over all samples."""
    args = parse_args()
    setup_logging(args.verbose)

    config = load_config(args.config)
    random_seed = int(config["runtime"]["random_seed"])
    random.seed(random_seed)
    np.random.seed(random_seed)

    model_cfg = config["model"]
    master_cfg = config["nodes"]["master"]
    worker_cfg = config["nodes"]["worker"]
    scheduler_cfg = config["scheduler"]
    eval_cfg = config["evaluation"]

    llm = load_model(
        model_path=str(model_cfg["path"]),
        n_threads=int(master_cfg["n_threads"]),
        n_ctx=int(model_cfg["n_ctx"]),
        logprobs=int(model_cfg["logprobs"]),
    )
    nlp = load_nlp_pipeline()

    samples = load_dataset(args.dataset)
    scheduler = NaiveParallelScheduler()
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    request_timeout_s = float(scheduler_cfg.get("request_timeout_s", 30.0))
    probe_tokens = int(scheduler_cfg.get("probe_tokens", 20))

    run_latencies: List[float] = []
    all_pro_scores: List[float] = []
    correctness_scores: List[float] = []

    for sample in samples:
        sample_start = time.perf_counter()
        prompt = str(sample.get("original_prompt", "")).strip()
        references = [str(item) for item in sample.get("ground_truth", [])]

        step_latencies_ms: Dict[str, int] = {
            "decomposition": 0,
            "scheduling": 0,
            "probe": 0,
            "inference": 0,
            "attribution": 0,
            "aggregation": 0,
            "metrics": 0,
        }

        decomposition_start = time.perf_counter()
        subtasks = decompose_prompt(prompt=prompt, llm=llm)
        step_latencies_ms["decomposition"] = int((time.perf_counter() - decomposition_start) * 1000)

        scheduling_start = time.perf_counter()
        scheduled = scheduler.schedule(subtasks)
        step_latencies_ms["scheduling"] = int((time.perf_counter() - scheduling_start) * 1000)

        pro_scores_by_id: Dict[int, float] = {}
        for subtask in scheduled:
            probe_start = time.perf_counter()
            probe = generate(
                llm=llm,
                prompt=str(subtask["text"]),
                max_tokens=min(probe_tokens, int(model_cfg["max_tokens"])),
                temperature=float(model_cfg["temperature"]),
                top_p=float(model_cfg["top_p"]),
                logprobs=int(model_cfg["logprobs"]),
                max_inference_ms=int(model_cfg["max_inference_ms"]),
            )
            step_latencies_ms["probe"] += int((time.perf_counter() - probe_start) * 1000)
            score = compute_pro_score(probe.get("pro_logprobs") or probe.get("logprobs", []), adaptive_k=True)
            pro_scores_by_id[int(subtask["id"])] = score
            all_pro_scores.append(score)

        results_by_id: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(scheduled))) as executor:
            future_map = {
                executor.submit(
                    execute_subtask,
                    subtask,
                    llm,
                    nlp,
                    model_cfg,
                    worker_cfg,
                    request_timeout_s,
                ): int(subtask["id"])
                for subtask in scheduled
            }
            for future in as_completed(future_map):
                subtask_id = future_map[future]
                results_by_id[subtask_id] = future.result()

        ordered_ids = [int(item["id"]) for item in scheduled]
        ordered_results = [results_by_id[subtask_id] for subtask_id in ordered_ids]
        step_latencies_ms["inference"] = sum(int(item.get("latency_ms", 0)) for item in ordered_results)
        outputs = [str(item.get("output", "")) for item in ordered_results]
        uncertainty_scores = [float(pro_scores_by_id.get(subtask_id, float("nan"))) for subtask_id in ordered_ids]
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

        correctness_value = rouge["rouge1"] if not np.isnan(rouge["rouge1"]) else 0.0
        correctness_scores.append(correctness_value)

        latency_ms = int((time.perf_counter() - sample_start) * 1000)
        run_latencies.append(latency_ms)

        node_a_latency = sum(int(item.get("latency_ms", 0)) for item in ordered_results if item.get("effective_node") == "node_a")
        node_b_latency = sum(int(item.get("latency_ms", 0)) for item in ordered_results if item.get("effective_node") == "node_b")

        record = {
            "id": sample.get("id"),
            "pipeline": "naive",
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
            "step_latencies_ms": step_latencies_ms,
            "node_latencies_ms": {"node_a": node_a_latency, "node_b": node_b_latency},
            "fallback": any(bool(item.get("fallback", False)) for item in ordered_results),
            "config_hash": config_hash(config),
            "model_path": model_cfg["path"],
            "quantization": "Q4_K_M",
        }
        append_jsonl(output_path, record)
        LOGGER.info("Processed sample id=%s latency=%sms", sample.get("id"), latency_ms)

    latency_stats = compute_latency_stats(run_latencies)
    erce = compute_erce(all_pro_scores, correctness_scores, n_bins=int(eval_cfg["n_bins"]))
    auroc = compute_auroc(all_pro_scores, correctness_scores)

    LOGGER.info("===== Naive Pipeline Summary =====")
    LOGGER.info("Samples: %s", len(samples))
    LOGGER.info("Latency mean=%.2f p95=%.2f p99=%.2f", latency_stats["mean"], latency_stats["p95"], latency_stats["p99"])
    LOGGER.info("Mean PRO score: %.4f", float(np.nanmean(np.array(all_pro_scores, dtype=float))))
    LOGGER.info("ERCE: %.4f", erce)
    LOGGER.info("AUROC: %.4f", auroc)


if __name__ == "__main__":
    main()
