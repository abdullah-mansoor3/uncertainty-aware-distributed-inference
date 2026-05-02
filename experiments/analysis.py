"""Experiment Results Analysis for ParallelPrompt (2nd run).

Pipelines: Serial | Naive MPI | Adaptive MPI
The outputs and plots are designed for paper-ready comparisons and ablations.
"""

import json
import math
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

random.seed(42)

SERIAL_PATH = "./results/results_serial_parallelprompt.jsonl"
NAIVE_PATH = "./results/results_naive_mpi_parallelprompt_2ndrun.jsonl"
ADAPTIVE_PATH = "./results/results_adaptive_mpi_parallelprompt_2ndrun.jsonl"

OUT_DIR = Path("./results/outputs/analysis_parallelprompt_2ndrun")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 80


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def is_valid_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


serial_rows = load_jsonl(SERIAL_PATH)
naive_rows = load_jsonl(NAIVE_PATH)
adaptive_rows = load_jsonl(ADAPTIVE_PATH)

datasets = {
    "Serial": serial_rows,
    "Naive": naive_rows,
    "Adaptive": adaptive_rows,
}

COLORS = {"Serial": "#2196F3", "Naive": "#FF9800", "Adaptive": "#4CAF50"}

print(SEP)
print("EXPERIMENT RESULTS ANALYSIS")
print("ParallelPrompt 2nd Run — Serial vs Naive MPI vs Adaptive MPI")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA QUALITY & BASIC SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

summary_rows = []
quality_rows = []

for label, rows in datasets.items():
    df = pd.json_normalize(rows)
    n_samples = len(df)
    n_subtasks = df["subtasks"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    latency = pd.to_numeric(df.get("latency_ms"), errors="coerce")
    merged_ok = df["merged_output"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)

    summary_rows.append(
        {
            "Pipeline": label,
            "Samples": n_samples,
            "Mean Subtasks": float(np.mean(n_subtasks)) if n_samples else float("nan"),
            "Median Latency (ms)": float(np.nanpercentile(latency, 50)) if n_samples else float("nan"),
            "Mean Latency (ms)": float(np.nanmean(latency)) if n_samples else float("nan"),
            "P95 Latency (ms)": float(np.nanpercentile(latency, 95)) if n_samples else float("nan"),
            "P99 Latency (ms)": float(np.nanpercentile(latency, 99)) if n_samples else float("nan"),
            "Merged Output Empty %": float((~merged_ok).mean() * 100) if n_samples else float("nan"),
        }
    )

    correctness = df.get("correctness", pd.Series([{}] * n_samples))
    bert_nan = sum(
        1
        for r in rows
        if isinstance(r.get("correctness", {}).get("bert"), float)
        and math.isnan(r.get("correctness", {}).get("bert"))
    )
    pro_vals = [v for r in rows for v in (r.get("uncertainty_scores") or []) if v is not None]
    pro_valid = [v for v in pro_vals if not (isinstance(v, float) and math.isnan(v))]
    quality_rows.append(
        {
            "Pipeline": label,
            "Valid PRO %": (len(pro_valid) / max(len(pro_vals), 1)) * 100,
            "BERT NaN %": (bert_nan / max(n_samples, 1)) * 100,
            "Has step_latencies_ms %": (df.get("step_latencies_ms").apply(lambda x: isinstance(x, dict)).mean() * 100)
            if "step_latencies_ms" in df.columns
            else 0.0,
        }
    )

summary_df = pd.DataFrame(summary_rows)
quality_df = pd.DataFrame(quality_rows)
summary_df.to_csv(OUT_DIR / "summary_overview.csv", index=False)
quality_df.to_csv(OUT_DIR / "data_quality_checks.csv", index=False)

print("\nSummary Overview:")
print(summary_df.to_string(index=False))
print("\nData Quality Checks:")
print(quality_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 2. CORRECTNESS METRICS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def safe_mean_corr(rows: List[Dict[str, Any]], metric: str) -> float:
    vals = [
        r.get("correctness", {}).get(metric)
        for r in rows
        if r.get("correctness")
        and r.get("correctness", {}).get(metric) is not None
        and not (isinstance(r.get("correctness", {}).get(metric), float) and math.isnan(r.get("correctness", {}).get(metric)))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def pct_zero(rows: List[Dict[str, Any]], metric: str) -> float:
    vals = [
        r.get("correctness", {}).get(metric)
        for r in rows
        if r.get("correctness")
        and r.get("correctness", {}).get(metric) is not None
        and not (isinstance(r.get("correctness", {}).get(metric), float) and math.isnan(r.get("correctness", {}).get(metric)))
    ]
    if not vals:
        return float("nan")
    return float(sum(1 for v in vals if v == 0) / len(vals) * 100)


correctness_rows = []
for label, rows in datasets.items():
    correctness_rows.append(
        {
            "Pipeline": label,
            "ROUGE-1": safe_mean_corr(rows, "rouge1"),
            "ROUGE-L": safe_mean_corr(rows, "rougeL"),
            "METEOR": safe_mean_corr(rows, "meteor"),
            "BLEU": safe_mean_corr(rows, "bleu"),
            "BERTScore": safe_mean_corr(rows, "bert"),
            "ROUGE-1 Zero %": pct_zero(rows, "rouge1"),
            "METEOR Zero %": pct_zero(rows, "meteor"),
        }
    )

correctness_df = pd.DataFrame(correctness_rows)
correctness_df.to_csv(OUT_DIR / "correctness_summary.csv", index=False)
print("\nCorrectness Summary:")
print(correctness_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 3. LATENCY & STEP LATENCY BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

latency_rows = []
step_rows = []

for label, rows in datasets.items():
    lats = np.array([safe_float(r.get("latency_ms")) for r in rows], dtype=float)
    node_a = np.array([safe_float(r.get("node_latencies_ms", {}).get("node_a")) for r in rows], dtype=float)
    node_b = np.array([safe_float(r.get("node_latencies_ms", {}).get("node_b")) for r in rows], dtype=float)

    latency_rows.append(
        {
            "Pipeline": label,
            "N": len(lats),
            "Mean (ms)": float(np.nanmean(lats)),
            "Median (ms)": float(np.nanpercentile(lats, 50)),
            "P95 (ms)": float(np.nanpercentile(lats, 95)),
            "P99 (ms)": float(np.nanpercentile(lats, 99)),
            "NodeA Mean (ms)": float(np.nanmean(node_a)),
            "NodeB Mean (ms)": float(np.nanmean(node_b)),
        }
    )

    steps_agg: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        for k, v in (r.get("step_latencies_ms") or {}).items():
            if is_valid_number(v):
                steps_agg[k].append(float(v))
    for step_name, values in steps_agg.items():
        if not values:
            continue
        step_rows.append(
            {
                "Pipeline": label,
                "Step": step_name,
                "Mean (ms)": float(np.mean(values)),
                "P95 (ms)": float(np.percentile(values, 95)),
                "Max (ms)": float(np.max(values)),
            }
        )

latency_df = pd.DataFrame(latency_rows)
step_df = pd.DataFrame(step_rows)
latency_df.to_csv(OUT_DIR / "latency_summary.csv", index=False)
step_df.to_csv(OUT_DIR / "step_latency_summary.csv", index=False)

print("\nLatency Summary:")
print(latency_df.to_string(index=False))
if not step_df.empty:
    print("\nStep Latency Summary (Top steps by mean):")
    top_steps = step_df.sort_values(["Pipeline", "Mean (ms)"], ascending=[True, False]).groupby("Pipeline").head(5)
    print(top_steps.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. ROUTING & LOAD DISTRIBUTION (MPI FOCUS)
# ─────────────────────────────────────────────────────────────────────────────

routing_rows = []
worker_latency_rows = []
roundtrip_rows = []

for label, rows in datasets.items():
    node_counts = Counter()
    proposed_counts = Counter()
    overrides = 0
    total_routes = 0

    for r in rows:
        routing = r.get("routing", {}) or {}
        proposed = r.get("routing_proposed_by_pro", {}) or {}
        for k, v in routing.items():
            node_counts[v] += 1
            total_routes += 1
            if proposed and proposed.get(k) and proposed.get(k) != v:
                overrides += 1
        for _, v in proposed.items():
            proposed_counts[v] += 1

        for item in r.get("per_subtask_latencies_ms", []) or []:
            wrank = item.get("worker_rank")
            rt = safe_float(item.get("round_trip_ms"))
            inf = safe_float(item.get("inference_ms"))
            if is_valid_number(rt):
                roundtrip_rows.append({"Pipeline": label, "worker_rank": wrank, "round_trip_ms": rt, "inference_ms": inf})
            if wrank is not None and is_valid_number(rt):
                worker_latency_rows.append({"Pipeline": label, "worker_rank": int(wrank), "round_trip_ms": rt})

    routing_rows.append(
        {
            "Pipeline": label,
            "Total Subtasks": total_routes,
            "Node A %": (node_counts.get("node_a", 0) / max(total_routes, 1)) * 100,
            "Node B %": (node_counts.get("node_b", 0) / max(total_routes, 1)) * 100,
            "Overrides %": (overrides / max(total_routes, 1)) * 100 if total_routes else float("nan"),
        }
    )

routing_df = pd.DataFrame(routing_rows)
roundtrip_df = pd.DataFrame(roundtrip_rows)
worker_latency_df = pd.DataFrame(worker_latency_rows)
routing_df.to_csv(OUT_DIR / "routing_summary.csv", index=False)
roundtrip_df.to_csv(OUT_DIR / "per_subtask_latencies.csv", index=False)

if not worker_latency_df.empty:
    worker_summary = worker_latency_df.groupby(["Pipeline", "worker_rank"]).agg(
        mean_round_trip_ms=("round_trip_ms", "mean"),
        p95_round_trip_ms=("round_trip_ms", lambda x: np.percentile(x, 95)),
        n=("round_trip_ms", "count"),
    )
    worker_summary.reset_index().to_csv(OUT_DIR / "worker_rank_latency_summary.csv", index=False)

print("\nRouting Summary:")
print(routing_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 5. DECOMPOSITION GATE & ADAPTIVE-SPECIFIC FLAGS
# ─────────────────────────────────────────────────────────────────────────────

adaptive_flags = []
for label, rows in datasets.items():
    if label != "Adaptive":
        continue
    decomposed = [bool(r.get("decomposed")) for r in rows if "decomposed" in r]
    locally = [bool(r.get("locally_processed")) for r in rows if "locally_processed" in r]
    fallback = [bool(r.get("fallback")) for r in rows if "fallback" in r]
    fallback_serial = [bool(r.get("fallback_to_serial")) for r in rows if "fallback_to_serial" in r]

    adaptive_flags.append(
        {
            "Pipeline": label,
            "Decomposed %": (sum(decomposed) / max(len(decomposed), 1)) * 100 if decomposed else float("nan"),
            "Locally Processed %": (sum(locally) / max(len(locally), 1)) * 100 if locally else float("nan"),
            "Fallback %": (sum(fallback) / max(len(fallback), 1)) * 100 if fallback else float("nan"),
            "Fallback to Serial %": (sum(fallback_serial) / max(len(fallback_serial), 1)) * 100 if fallback_serial else float("nan"),
        }
    )

adaptive_flags_df = pd.DataFrame(adaptive_flags)
adaptive_flags_df.to_csv(OUT_DIR / "adaptive_flags_summary.csv", index=False)
if not adaptive_flags_df.empty:
    print("\nAdaptive Flags Summary:")
    print(adaptive_flags_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 6. UNCERTAINTY VS QUALITY & LATENCY
# ─────────────────────────────────────────────────────────────────────────────

uq_rows = []
for label, rows in datasets.items():
    for r in rows:
        pro_vals = [v for v in (r.get("uncertainty_scores") or []) if is_valid_number(v)]
        if not pro_vals:
            continue
        rouge1 = r.get("correctness", {}).get("rouge1")
        latency = r.get("latency_ms")
        uq_rows.append(
            {
                "Pipeline": label,
                "pro_mean": float(np.mean(pro_vals)),
                "rouge1": safe_float(rouge1),
                "latency_ms": safe_float(latency),
            }
        )

uq_df = pd.DataFrame(uq_rows)
uq_df.to_csv(OUT_DIR / "uncertainty_vs_quality.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 7. CALIBRATION SUMMARY (RUNNING METRICS)
# ─────────────────────────────────────────────────────────────────────────────

def latest_running_metric(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [safe_float(r.get(key)) for r in rows if is_valid_number(r.get(key))]
    return vals[-1] if vals else float("nan")


calib_rows = []
for label, rows in datasets.items():
    calib_rows.append(
        {
            "Pipeline": label,
            "ERCE (latest)": latest_running_metric(rows, "running_erce"),
            "AUROC (latest)": latest_running_metric(rows, "running_auroc"),
        }
    )

calib_df = pd.DataFrame(calib_rows)
calib_df.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
print("\nCalibration Summary:")
print(calib_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOTS (NON-REDUNDANT)
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerating plots...")

# Plot 1: Total latency distribution (box)
fig, ax = plt.subplots(figsize=(8, 4))
latency_data = {label: np.array([safe_float(r.get("latency_ms")) for r in rows]) / 1000 for label, rows in datasets.items()}
ax.boxplot([latency_data[l] for l in ["Serial", "Naive", "Adaptive"]], labels=["Serial", "Naive MPI", "Adaptive MPI"], patch_artist=True)
ax.set_ylabel("Latency (s)")
ax.set_title("Total Latency Distribution")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_latency_box.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Correctness vs latency tradeoff (ROUGE-1)
fig, ax = plt.subplots(figsize=(7, 5))
for label, rows in datasets.items():
    xs = [safe_float(r.get("latency_ms")) / 1000 for r in rows]
    ys = [safe_float(r.get("correctness", {}).get("rouge1")) for r in rows]
    ax.scatter(xs, ys, alpha=0.4, label=label, color=COLORS[label], edgecolors="white", s=35)
ax.set_xlabel("Latency (s)")
ax.set_ylabel("ROUGE-1")
ax.set_title("Latency vs ROUGE-1")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_latency_vs_rouge1.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 3: Step latency stack (mean) per pipeline
if not step_df.empty:
    pivot = step_df.pivot_table(index="Pipeline", columns="Step", values="Mean (ms)", aggfunc="mean").fillna(0)
    pivot = pivot.reindex(index=["Serial", "Naive", "Adaptive"]) if "Serial" in pivot.index else pivot
    pivot.plot(kind="bar", stacked=True, figsize=(10, 4))
    plt.ylabel("Mean step latency (ms)")
    plt.title("Mean Step Latency Composition")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_step_latency_stacked.png", dpi=150, bbox_inches="tight")
    plt.close()

# Plot 4: Routing distribution
fig, ax = plt.subplots(figsize=(6, 4))
node_a_pct = routing_df.set_index("Pipeline")["Node A %"]
node_b_pct = routing_df.set_index("Pipeline")["Node B %"]
idx = np.arange(len(node_a_pct))
ax.bar(idx, node_a_pct, label="node_a", color="#5DA5DA")
ax.bar(idx, node_b_pct, bottom=node_a_pct, label="node_b", color="#FAA43A")
ax.set_xticks(idx)
ax.set_xticklabels(node_a_pct.index)
ax.set_ylabel("Share of subtasks (%)")
ax.set_title("Routing Distribution")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_routing_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 5: PRO vs ROUGE-1 (all pipelines)
if not uq_df.empty:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in ["Serial", "Naive", "Adaptive"]:
        sub = uq_df[uq_df["Pipeline"] == label]
        if sub.empty:
            continue
        ax.scatter(sub["pro_mean"], sub["rouge1"], alpha=0.5, label=label, color=COLORS[label], edgecolors="white", s=35)
    ax.set_xlabel("Mean PRO score")
    ax.set_ylabel("ROUGE-1")
    ax.set_title("Uncertainty vs Correctness")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_pro_vs_rouge1.png", dpi=150, bbox_inches="tight")
    plt.close()

# Plot 6: Worker rank round-trip latency (MPI only)
if not worker_latency_df.empty:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in ["Naive", "Adaptive"]:
        sub = worker_latency_df[worker_latency_df["Pipeline"] == label]
        if sub.empty:
            continue
        ax.scatter(sub["worker_rank"], sub["round_trip_ms"], alpha=0.4, label=label, s=20)
    ax.set_xlabel("Worker rank")
    ax.set_ylabel("Round-trip latency (ms)")
    ax.set_title("Worker Rank Round-trip Latency")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_worker_rank_roundtrip.png", dpi=150, bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE FINAL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

comparison_df = summary_df.merge(correctness_df, on="Pipeline").merge(latency_df, on="Pipeline")
comparison_df.to_csv(OUT_DIR / "full_comparison.csv", index=False)
print("\nSaved outputs to:", OUT_DIR)
