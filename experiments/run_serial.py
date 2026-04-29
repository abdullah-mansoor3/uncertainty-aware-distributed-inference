"""Run the serial inference pipeline baseline.

Pipeline: data loading -> decomposition -> serial scheduling -> local inference ->
aggregation -> correctness metrics -> incremental JSONL logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import spacy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.aggregator import aggregate_attributions, merge_outputs
from src.modules.explanation import compute_local_attribution
from src.modules.inference import generate, load_model
from src.modules.uncertainty import compute_erce, compute_pro_score
from src.scheduler.serial import SerialScheduler
from src.utils.config_loader import config_hash, load_config
from src.utils.metrics import compute_bert_score, compute_bleu, compute_latency_stats, compute_meteor, compute_rouge

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for serial runner.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Run serial inference pipeline")
    parser.add_argument("--config", required=True, help="Path to cluster config YAML")
    parser.add_argument("--dataset", required=True, help="Path to processed dataset JSONL")
    parser.add_argument("--output", required=True, help="Path to output results JSONL")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Initialize logging.

    Args:
        verbose: Whether to use DEBUG level.

    Returns:
        None.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load processed JSONL dataset.

    Args:
        dataset_path: Path to dataset file.

    Returns:
        List of sample dictionaries.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file.

    Args:
        path: Output JSONL path.
        record: Record dictionary.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_nlp_pipeline() -> Any:
    """Load spaCy parser used for SyntaxShap-style attribution approximation.

    Returns:
        Loaded spaCy pipeline or lightweight fallback pipeline.
    """
    try:
        return spacy.load("en_core_web_sm")
    except Exception as error:
        LOGGER.warning("Failed to load en_core_web_sm (%s). Falling back to spacy.blank('en').", error)
        return spacy.blank("en")


def main() -> None:
    """Execute serial pipeline over all dataset samples.

    Returns:
        None.
    """
    args = parse_args()
    setup_logging(args.verbose)

    config = load_config(args.config)
    seed = int(config["runtime"]["random_seed"])
    random.seed(seed)
    np.random.seed(seed)

    model_config = config["model"]
    master_config = config["nodes"]["master"]
    scheduler_config = config["scheduler"]
    probe_tokens = int(scheduler_config.get("probe_tokens", 20))
    runtime_config = config.get("runtime", {})
    decomposition_use_llm = bool(runtime_config.get("decomposition_use_llm", False))
    enable_attribution = bool(runtime_config.get("enable_attribution", False))
    use_probe_for_pro = bool(runtime_config.get("use_probe_for_pro", False))

    available_cpus = os.cpu_count() or int(master_config["n_threads"])
    effective_threads = max(1, min(int(master_config["n_threads"]), available_cpus))
    if effective_threads != int(master_config["n_threads"]):
        LOGGER.warning(
            "Reducing n_threads from %s to %s based on available CPU cores.",
            master_config["n_threads"],
            effective_threads,
        )

    llm = load_model(
        model_path=model_config["path"],
        n_threads=effective_threads,
        n_ctx=int(model_config["n_ctx"]),
        logprobs=int(model_config["logprobs"]),
        logits_all=bool(model_config.get("logits_all", False)),
    )
    nlp = load_nlp_pipeline() if enable_attribution else None

    samples = load_dataset(args.dataset)
    scheduler = SerialScheduler()
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    run_latencies: List[float] = []
    pro_scores: List[float] = []
    correctness_scores: List[float] = []

    for sample in samples:
        sample_start = time.perf_counter()
        prompt = str(sample.get("original_prompt", "")).strip()
        if not prompt:
            LOGGER.warning("Skipping sample id=%s because prompt is empty.", sample.get("id"))
            continue

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
        subtasks = [{"id": 0, "text": prompt, "dependencies": [], "parallel_safe": False}]
        scheduling_start = time.perf_counter()
        scheduled_subtasks = scheduler.schedule(subtasks)
        step_latencies_ms["scheduling"] = int((time.perf_counter() - scheduling_start) * 1000)

        outputs: List[str] = []
        attribution_vectors: List[List[Dict[str, float]]] = []
        uncertainty_scores: List[float] = []
        node_latency_total = 0

        for subtask in scheduled_subtasks:
            probe_latency_ms = 0
            probe_logprobs: List[float] = []
            if use_probe_for_pro:
                probe_start = time.perf_counter()
                probe = generate(
                    llm=llm,
                    prompt=subtask["text"],
                    max_tokens=min(probe_tokens, int(model_config["max_tokens"])),
                    temperature=float(model_config["temperature"]),
                    top_p=float(model_config["top_p"]),
                    logprobs=int(model_config["logprobs"]),
                    max_inference_ms=int(model_config["max_inference_ms"]),
                )
                probe_latency_ms = int(probe.get("latency_ms", 0))
                probe_logprobs = probe.get("pro_logprobs") or probe.get("logprobs", [])
                step_latencies_ms["probe"] += int((time.perf_counter() - probe_start) * 1000)

            inference_start = time.perf_counter()
            generation = generate(
                llm=llm,
                prompt=subtask["text"],
                max_tokens=int(model_config["max_tokens"]),
                temperature=float(model_config["temperature"]),
                top_p=float(model_config["top_p"]),
                logprobs=int(model_config["logprobs"]),
                max_inference_ms=int(model_config["max_inference_ms"]),
            )
            step_latencies_ms["inference"] += int((time.perf_counter() - inference_start) * 1000)
            pro_score = compute_pro_score(
                probe_logprobs if use_probe_for_pro else (generation.get("pro_logprobs") or generation.get("logprobs", [])),
                adaptive_k=True,
            )
            outputs.append(generation["text"])
            local_attribution: List[Dict[str, float]] = []
            if enable_attribution and nlp is not None:
                # SyntaxShap is intentionally approximate here for CPU feasibility.
                try:
                    attribution_start = time.perf_counter()
                    local_attribution = compute_local_attribution(
                        subtask=subtask["text"],
                        output=generation["text"],
                        llm=llm,
                        nlp=nlp,
                    )
                    step_latencies_ms["attribution"] += int((time.perf_counter() - attribution_start) * 1000)
                except Exception as error:
                    LOGGER.warning("Local attribution failed for subtask id=%s: %s", subtask.get("id"), error)
            attribution_vectors.append(local_attribution)
            uncertainty_scores.append(pro_score)
            pro_scores.append(pro_score)
            node_latency_total += probe_latency_ms + int(generation.get("latency_ms", 0))

        aggregation_start = time.perf_counter()
        merged_output = outputs[0] if outputs else ""
        global_attribution = aggregate_attributions(attribution_vectors, uncertainty_scores)
        step_latencies_ms["aggregation"] = int((time.perf_counter() - aggregation_start) * 1000)

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

        record = {
            "id": sample.get("id"),
            "pipeline": "serial",
            "original_prompt": prompt,
            "subtasks": [item["text"] for item in scheduled_subtasks],
            "routing": {f"subtask_{index}": "node_a" for index, _ in enumerate(scheduled_subtasks)},
            "uncertainty_scores": uncertainty_scores,
            "outputs": outputs,
            "merged_output": merged_output,
            "attribution_vectors": attribution_vectors,
            "global_attribution": global_attribution,
            "correctness": {
                "rouge1": rouge["rouge1"],
                "rougeL": rouge["rougeL"],
                "meteor": meteor,
                "bleu": bleu,
                "bert": bert,
            },
            "latency_ms": latency_ms,
            "step_latencies_ms": step_latencies_ms,
            "node_latencies_ms": {"node_a": node_latency_total, "node_b": 0},
            "config_hash": config_hash(config),
            "model_path": model_config["path"],
            "quantization": "Q4_K_M",
        }
        append_jsonl(output_path, record)
        LOGGER.info("Processed sample id=%s latency=%sms", sample.get("id"), latency_ms)

    latency_stats = compute_latency_stats(run_latencies)
    erce_value = compute_erce(pro_scores, correctness_scores, n_bins=int(config["evaluation"]["n_bins"]))

    LOGGER.info("===== Serial Pipeline Summary =====")
    LOGGER.info("Samples: %s", len(samples))
    LOGGER.info(
        "decomposition_use_llm=%s enable_attribution=%s use_probe_for_pro=%s",
        decomposition_use_llm,
        enable_attribution,
        use_probe_for_pro,
    )
    LOGGER.info("Latency mean=%.2f p95=%.2f p99=%.2f", latency_stats["mean"], latency_stats["p95"], latency_stats["p99"])
    LOGGER.info("Mean PRO score: %.4f", float(np.nanmean(np.array(pro_scores, dtype=float))))
    LOGGER.info("ERCE: %.4f", erce_value)


if __name__ == "__main__":
    main()
