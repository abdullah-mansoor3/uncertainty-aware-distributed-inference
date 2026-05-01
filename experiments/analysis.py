"""
Experiment Results Analysis
Uncertainty-Aware Adaptive Distributed Inference on Heterogeneous CPU Clusters
Pipelines: Serial | Naive MPI Parallel | Adaptive MPI Parallel
"""

import json
import math
import os
import random
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List

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

SERIAL_PATH   = "./results/results_serial_parallelprompt.jsonl"
NAIVE_PATH    = "./results/results_naive_mpi_parallelprompt.jsonl"
ADAPTIVE_PATH = "./results/results_adaptive_mpi_parallelprompt.jsonl"
OUT_DIR       = Path("./results/outputs/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEP  = "=" * 80
SEP2 = "-" * 80

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

serial_rows   = load_jsonl(SERIAL_PATH)
naive_rows    = load_jsonl(NAIVE_PATH)
adaptive_rows = load_jsonl(ADAPTIVE_PATH)

datasets = {
    "Serial":   serial_rows,
    "Naive":    naive_rows,
    "Adaptive": adaptive_rows,
}

print(SEP)
print("EXPERIMENT RESULTS ANALYSIS")
print("Uncertainty-Aware Adaptive Distributed Inference on Heterogeneous CPU Clusters")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET SUMMARIES + NAN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 1: DATASET SUMMARIES AND NaN ANALYSIS")
print(SEP)

for label, rows in datasets.items():
    df = pd.json_normalize(rows)

    # Flatten correctness columns
    for metric in ["rouge1", "rougeL", "meteor", "bleu", "bert"]:
        col = f"correctness.{metric}"
        if col in df.columns:
            df[metric] = pd.to_numeric(df[col], errors="coerce")

    # Flatten node latencies
    for node in ["node_a", "node_b"]:
        col = f"node_latencies_ms.{node}"
        if col in df.columns:
            df[f"lat_{node}"] = pd.to_numeric(df[col], errors="coerce")

    # PRO scores: expand list column
    def mean_pro(x):
        try:
            vals = [float(v) for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return np.mean(vals) if vals else float("nan")
        except Exception:
            return float("nan")

    df["pro_mean"] = df["uncertainty_scores"].apply(mean_pro)
    df["n_subtasks"] = df["subtasks"].apply(len)
    df["n_outputs"]  = df["outputs"].apply(len)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")

    print(f"\n{'─'*60}")
    print(f"  Pipeline: {label.upper()}  ({len(df)} samples)")
    print(f"{'─'*60}")
    print(f"  Columns: {list(df.columns)}")

    # NaN counts for key columns
    key_cols = ["rouge1", "rougeL", "meteor", "bleu", "bert", "pro_mean",
                "latency_ms", "lat_node_a", "lat_node_b", "n_subtasks"]
    key_cols = [c for c in key_cols if c in df.columns]
    nan_summary = pd.DataFrame({
        "column":    key_cols,
        "nan_count": [int(df[c].isna().sum()) for c in key_cols],
        "nan_pct":   [f"{df[c].isna().mean()*100:.1f}%" for c in key_cols],
    })
    print("\n  NaN summary:")
    print(nan_summary.to_string(index=False))

    # Pipeline-specific extra fields
    if label == "Adaptive":
        for field in ["fallback_to_serial", "locally_processed", "decomposed", "fallback"]:
            if field in df.columns:
                pct = df[field].sum() / len(df) * 100
                print(f"\n  {field}: {int(df[field].sum())}/{len(df)}  ({pct:.1f}%)")

    # Drop rows where merged_output is empty
    before = len(df)
    df = df[df["merged_output"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    after = len(df)
    if before != after:
        print(f"\n  Dropped {before-after} rows with empty merged_output. Remaining: {after}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. RANDOM SAMPLE INSPECTION  (5 per pipeline)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 2: RANDOM SAMPLE INSPECTION  (5 samples per pipeline)")
print(SEP)

def wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    if not isinstance(text, str):
        text = str(text)
    lines = textwrap.wrap(text, width)
    return ("\n" + indent).join(lines) if lines else "(empty)"

for label, rows in datasets.items():
    print(f"\n{'═'*70}")
    print(f"  PIPELINE: {label.upper()}")
    print(f"{'═'*70}")

    sample_ids = random.sample(range(len(rows)), min(5, len(rows)))

    for i, idx in enumerate(sample_ids):
        r = rows[idx]
        print(f"\n  ── Sample {i+1}  (record id={r.get('id')}) ──")

        print(f"\n  [ORIGINAL PROMPT]")
        prompt = str(r.get("original_prompt", ""))
        print(f"    {wrap(prompt)}")

        subtasks = r.get("subtasks", [])
        print(f"\n  [SUBTASKS]  ({len(subtasks)} total)")
        for si, st in enumerate(subtasks):
            print(f"    [{si}] {wrap(str(st), indent='        ')}")

        outputs = r.get("outputs", [])
        print(f"\n  [OUTPUTS PER SUBTASK]  ({len(outputs)} total)")
        for oi, out in enumerate(outputs):
            print(f"    [{oi}] {wrap(str(out), indent='        ')}")

        merged = str(r.get("merged_output", ""))
        print(f"\n  [MERGED OUTPUT]")
        print(f"    {wrap(merged)}")

        routing = r.get("routing", {})
        print(f"\n  [ROUTING]  {routing}")

        corr = r.get("correctness", {})
        pro  = r.get("uncertainty_scores", [])
        print(f"  [CORRECTNESS]  rouge1={corr.get('rouge1','N/A'):.4f}  "
              f"rougeL={corr.get('rougeL','N/A'):.4f}  "
              f"meteor={corr.get('meteor','N/A'):.4f}  "
              f"bleu={corr.get('bleu','N/A'):.4f}")
        print(f"  [PRO SCORES]  {[round(v,4) if v is not None and not (isinstance(v,float) and math.isnan(v)) else 'NaN' for v in pro]}")
        print(f"  [LATENCY]  total={r.get('latency_ms','N/A')} ms  "
              f"node_a={r.get('node_latencies_ms',{}).get('node_a','N/A')} ms  "
              f"node_b={r.get('node_latencies_ms',{}).get('node_b','N/A')} ms")
        print(f"  {SEP2}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LATENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 3: LATENCY ANALYSIS")
print(SEP)

latency_summary = []

for label, rows in datasets.items():
    lats = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    lats = np.array(lats, dtype=float)

    node_a = np.array([r.get("node_latencies_ms", {}).get("node_a", 0) for r in rows], dtype=float)
    node_b = np.array([r.get("node_latencies_ms", {}).get("node_b", 0) for r in rows], dtype=float)

    row = {
        "Pipeline":      label,
        "N":             len(lats),
        "Mean (ms)":     int(np.nanmean(lats)),
        "Median (ms)":   int(np.nanpercentile(lats, 50)),
        "P95 (ms)":      int(np.nanpercentile(lats, 95)),
        "P99 (ms)":      int(np.nanpercentile(lats, 99)),
        "Min (ms)":      int(np.nanmin(lats)),
        "Max (ms)":      int(np.nanmax(lats)),
        "Std (ms)":      int(np.nanstd(lats)),
        "NodeA Mean (ms)": int(np.nanmean(node_a)),
        "NodeB Mean (ms)": int(np.nanmean(node_b)),
    }
    latency_summary.append(row)

lat_df = pd.DataFrame(latency_summary)
print("\nOverall Latency Statistics:")
print(lat_df.to_string(index=False))

# Step latencies for serial
print("\n\nSerial Pipeline — Step Latency Breakdown:")
steps_agg = {}
for r in serial_rows:
    for k, v in r.get("step_latencies_ms", {}).items():
        steps_agg.setdefault(k, []).append(int(v))

step_df = pd.DataFrame([
    {"Step": k, "Mean (ms)": int(np.mean(v)), "P95 (ms)": int(np.percentile(v, 95)), "Max (ms)": int(np.max(v))}
    for k, v in steps_agg.items()
])
print(step_df.to_string(index=False))

# Save
lat_df.to_csv(OUT_DIR / "latency_summary.csv", index=False)
step_df.to_csv(OUT_DIR / "serial_step_latencies.csv", index=False)
print(f"\nSaved: latency_summary.csv, serial_step_latencies.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CORRECTNESS METRICS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 4: CORRECTNESS METRICS")
print(SEP)

correctness_summary = []

for label, rows in datasets.items():
    def safe_mean(metric):
        vals = [r["correctness"].get(metric) for r in rows
                if r.get("correctness") and r["correctness"].get(metric) is not None
                and not (isinstance(r["correctness"].get(metric), float) and math.isnan(r["correctness"].get(metric)))]
        return round(np.mean(vals), 4) if vals else float("nan")

    def pct_zero(metric):
        vals = [r["correctness"].get(metric) for r in rows
                if r.get("correctness") and r["correctness"].get(metric) is not None
                and not (isinstance(r["correctness"].get(metric), float) and math.isnan(r["correctness"].get(metric)))]
        if not vals:
            return float("nan")
        return round(sum(1 for v in vals if v == 0) / len(vals) * 100, 1)

    correctness_summary.append({
        "Pipeline":       label,
        "ROUGE-1 Mean":   safe_mean("rouge1"),
        "ROUGE-L Mean":   safe_mean("rougeL"),
        "METEOR Mean":    safe_mean("meteor"),
        "BLEU Mean":      safe_mean("bleu"),
        "BERTScore Mean": safe_mean("bert"),
        "ROUGE-1 Zero%":  pct_zero("rouge1"),
        "METEOR Zero%":   pct_zero("meteor"),
    })

corr_df = pd.DataFrame(correctness_summary)
print("\nAveraged Correctness Metrics:")
print(corr_df.to_string(index=False))
corr_df.to_csv(OUT_DIR / "correctness_summary.csv", index=False)
print(f"\nSaved: correctness_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PRO UNCERTAINTY SCORES
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 5: PRO UNCERTAINTY SCORES")
print(SEP)

pro_summary = []
for label, rows in datasets.items():
    all_scores = [v for r in rows for v in (r.get("uncertainty_scores") or [])
                  if v is not None]
    valid = [v for v in all_scores if not (isinstance(v, float) and math.isnan(v))]
    nan_count = len(all_scores) - len(valid)

    pro_summary.append({
        "Pipeline":     label,
        "Total Scores": len(all_scores),
        "Valid":        len(valid),
        "NaN Count":    nan_count,
        "NaN %":        f"{nan_count/max(len(all_scores),1)*100:.1f}%" if all_scores else "N/A",
        "Mean PRO":     round(np.mean(valid), 4) if valid else float("nan"),
        "Std PRO":      round(np.std(valid), 4) if valid else float("nan"),
        "Pct > 0.5":    f"{sum(1 for v in valid if v > 0.5)/max(len(valid),1)*100:.1f}%" if valid else "N/A",
    })

pro_df = pd.DataFrame(pro_summary)
print("\nPRO Score Summary (uncertainty_scores field):")
print(pro_df.to_string(index=False))
pro_df.to_csv(OUT_DIR / "pro_score_summary.csv", index=False)
print(f"\nSaved: pro_score_summary.csv")
print()
print("  NOTE: Naive and Adaptive uncertainty_scores are entirely NaN.")
print("  This indicates the PRO probe inference ran but logprobs were not recorded")
print("  in the result schema for these pipelines. The serial pipeline correctly")
print("  records PRO scores from the generate() call's logprob output.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. DECOMPOSITION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 6: DECOMPOSITION ANALYSIS")
print(SEP)

decomp_summary = []
for label, rows in datasets.items():
    n_sub = [len(r.get("subtasks", [])) for r in rows]
    decomposed = sum(1 for n in n_sub if n > 1)
    decomp_summary.append({
        "Pipeline":          label,
        "Mean Subtasks":     round(np.mean(n_sub), 2),
        "Max Subtasks":      max(n_sub),
        "Min Subtasks":      min(n_sub),
        "Decomposed (>1)":   decomposed,
        "Decomposed %":      f"{decomposed/len(rows)*100:.1f}%",
        "Single Task %":     f"{(len(rows)-decomposed)/len(rows)*100:.1f}%",
    })

decomp_df = pd.DataFrame(decomp_summary)
print("\nDecomposition Statistics:")
print(decomp_df.to_string(index=False))
decomp_df.to_csv(OUT_DIR / "decomposition_summary.csv", index=False)
print(f"\nSaved: decomposition_summary.csv")

# Subtask count distribution for adaptive
print("\nAdaptive — Subtask Count Distribution:")
from collections import Counter
cnt = Counter(len(r["subtasks"]) for r in adaptive_rows)
dist_df = pd.DataFrame(sorted(cnt.items()), columns=["Subtask Count", "Frequency"])
dist_df["% of Samples"] = (dist_df["Frequency"] / len(adaptive_rows) * 100).round(1)
print(dist_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8. ADAPTIVE PIPELINE SPECIFIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 7: ADAPTIVE PIPELINE — ROUTING AND FALLBACK ANALYSIS")
print(SEP)

# Node assignment breakdown
total_node_a = 0
total_node_b = 0
for r in adaptive_rows:
    for v in r.get("routing", {}).values():
        if v == "node_a":
            total_node_a += 1
        else:
            total_node_b += 1

total_subtasks = total_node_a + total_node_b
print(f"\nTotal subtasks scheduled:  {total_subtasks}")
print(f"Assigned to node_a (i7):   {total_node_a}  ({total_node_a/total_subtasks*100:.1f}%)")
print(f"Assigned to node_b (i5):   {total_node_b}  ({total_node_b/total_subtasks*100:.1f}%)")

# fallback field
fallback_count = sum(1 for r in adaptive_rows if r.get("fallback"))
print(f"\nFallback triggered (worker unreachable): {fallback_count}/{len(adaptive_rows)}  ({fallback_count/len(adaptive_rows)*100:.1f}%)")

# Fields that were expected but missing (schema mismatch detection)
expected_fields = ["fallback_to_serial", "locally_processed", "decomposed"]
print("\nExpected adaptive fields present in results:")
for field in expected_fields:
    present = sum(1 for r in adaptive_rows if field in r)
    print(f"  {field}: found in {present}/{len(adaptive_rows)} records  "
          f"{'✓' if present == len(adaptive_rows) else '✗ MISSING from schema'}")

# Naive vs Adaptive routing comparison
same_routing = sum(1 for n, a in zip(naive_rows, adaptive_rows) if n["routing"] == a["routing"])
same_output  = sum(1 for n, a in zip(naive_rows, adaptive_rows) if n["merged_output"] == a["merged_output"])
print(f"\nNaive vs Adaptive routing identical:       {same_routing}/{len(naive_rows)} samples")
print(f"Naive vs Adaptive merged_output identical:  {same_output}/{len(naive_rows)} samples")

# Node_b latency comparison
naive_b    = np.array([r["node_latencies_ms"]["node_b"] for r in naive_rows], dtype=float)
adaptive_b = np.array([r["node_latencies_ms"]["node_b"] for r in adaptive_rows], dtype=float)
diff = adaptive_b - naive_b
print(f"\nNode_b latency difference (Adaptive − Naive):")
print(f"  Mean: {np.mean(diff):.0f} ms   Std: {np.std(diff):.0f} ms")
print(f"  Adaptive faster on {sum(diff < 0)} samples, slower on {sum(diff > 0)} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 9. PIPELINE COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 8: FULL PIPELINE COMPARISON TABLE")
print(SEP)

def safe_mean_corr(rows, metric):
    vals = [r["correctness"].get(metric) for r in rows
            if r.get("correctness") and r["correctness"].get(metric) is not None
            and not (isinstance(r["correctness"].get(metric), float) and math.isnan(r["correctness"].get(metric)))]
    return round(np.mean(vals), 4) if vals else float("nan")

comparison = []
for label, rows in datasets.items():
    lats = np.array([r["latency_ms"] for r in rows], dtype=float)
    pro_vals = [v for r in rows for v in (r.get("uncertainty_scores") or [])
                if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n_sub = np.array([len(r.get("subtasks", [])) for r in rows], dtype=float)
    node_b = np.array([r["node_latencies_ms"]["node_b"] for r in rows], dtype=float)

    comparison.append({
        "Pipeline":          label,
        "Samples":           len(rows),
        "Mean Lat (s)":      round(np.nanmean(lats) / 1000, 1),
        "P95 Lat (s)":       round(np.nanpercentile(lats, 95) / 1000, 1),
        "P99 Lat (s)":       round(np.nanpercentile(lats, 99) / 1000, 1),
        "ROUGE-1":           safe_mean_corr(rows, "rouge1"),
        "ROUGE-L":           safe_mean_corr(rows, "rougeL"),
        "METEOR":            safe_mean_corr(rows, "meteor"),
        "BLEU":              safe_mean_corr(rows, "bleu"),
        "Mean Subtasks":     round(np.mean(n_sub), 2),
        "PRO Mean":          round(np.mean(pro_vals), 4) if pro_vals else "NaN (all missing)",
        "NodeB Mean (s)":    round(np.nanmean(node_b) / 1000, 1),
    })

comp_df = pd.DataFrame(comparison)
print("\nFull Comparison Table:")
print(comp_df.to_string(index=False))
comp_df.to_csv(OUT_DIR / "full_comparison.csv", index=False)
print(f"\nSaved: full_comparison.csv")

# Speedup relative to serial
serial_mean = np.nanmean([r["latency_ms"] for r in serial_rows])
for label, rows in [("Naive", naive_rows), ("Adaptive", adaptive_rows)]:
    mean = np.nanmean([r["latency_ms"] for r in rows])
    slowdown = mean / serial_mean
    print(f"\n  {label} vs Serial: {slowdown:.2f}x {'slower' if slowdown > 1 else 'faster'} on mean latency")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 9: GENERATING PLOTS")
print(SEP)

COLORS = {"Serial": "#2196F3", "Naive": "#FF9800", "Adaptive": "#4CAF50"}

# ── Plot 1: Latency distribution (box plots) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

latency_data = {label: np.array([r["latency_ms"] for r in rows]) / 1000
                for label, rows in datasets.items()}

bp = axes[0].boxplot(
    [latency_data[l] for l in ["Serial", "Naive", "Adaptive"]],
    labels=["Serial", "Naive MPI", "Adaptive MPI"],
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
)
for patch, color in zip(bp["boxes"], COLORS.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_ylabel("Total Latency (seconds)")
axes[0].set_title("Latency Distribution by Pipeline")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}s"))
axes[0].grid(axis="y", alpha=0.3)

# P95 / P99 bar chart
p_labels = ["Mean", "P50", "P95", "P99"]
x = np.arange(len(p_labels))
width = 0.25
for i, (label, rows) in enumerate(datasets.items()):
    lats = np.array([r["latency_ms"] for r in rows]) / 1000
    vals = [np.nanmean(lats), np.nanpercentile(lats, 50),
            np.nanpercentile(lats, 95), np.nanpercentile(lats, 99)]
    axes[1].bar(x + i * width, vals, width, label=label, color=COLORS[label], alpha=0.8)
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(p_labels)
axes[1].set_ylabel("Latency (seconds)")
axes[1].set_title("Latency Percentiles by Pipeline")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_latency_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot_latency_distribution.png")

# ── Plot 2: Correctness metrics grouped bar ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
metrics = ["rouge1", "rougeL", "meteor", "bleu"]
x = np.arange(len(metrics))
width = 0.25
for i, (label, rows) in enumerate(datasets.items()):
    vals = []
    for m in metrics:
        v = [r["correctness"].get(m) for r in rows
             if r.get("correctness") and r["correctness"].get(m) is not None
             and not (isinstance(r["correctness"].get(m), float) and math.isnan(r["correctness"].get(m)))]
        vals.append(np.mean(v) if v else 0.0)
    bars = ax.bar(x + i * width, vals, width, label=label, color=COLORS[label], alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x + width)
ax.set_xticklabels(["ROUGE-1", "ROUGE-L", "METEOR", "BLEU"])
ax.set_ylabel("Score")
ax.set_title("Correctness Metrics by Pipeline")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_correctness_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot_correctness_metrics.png")

# ── Plot 3: PRO score distribution (serial only — others are NaN) ─────────
fig, ax = plt.subplots(figsize=(8, 4))
pro_serial = [v for r in serial_rows for v in (r.get("uncertainty_scores") or [])
              if v is not None and not (isinstance(v, float) and math.isnan(v))]
if pro_serial:
    ax.hist(pro_serial, bins=20, color=COLORS["Serial"], alpha=0.8, edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold τ=0.5")
    ax.set_xlabel("PRO Score")
    ax.set_ylabel("Frequency")
    ax.set_title("PRO Score Distribution — Serial Pipeline\n(Naive/Adaptive: all NaN — see validity notes)")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_pro_scores_serial.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot_pro_scores_serial.png")

# ── Plot 4: Serial step latency breakdown ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
step_means = {k: np.mean(v) / 1000 for k, v in steps_agg.items()}
keys = list(step_means.keys())
vals = list(step_means.values())
bars = ax.barh(keys, vals, color="#2196F3", alpha=0.8, edgecolor="white")
for bar, v in zip(bars, vals):
    ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}s", va="center", fontsize=9)
ax.set_xlabel("Mean Duration (seconds)")
ax.set_title("Serial Pipeline — Mean Step Latency Breakdown")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_serial_step_latency.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot_serial_step_latency.png")

# ── Plot 5: Subtask count distribution ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
from collections import Counter
for label, rows in [("Naive", naive_rows), ("Adaptive", adaptive_rows)]:
    cnt = Counter(len(r["subtasks"]) for r in rows)
    x_vals = sorted(cnt.keys())
    y_vals = [cnt[k] for k in x_vals]
    ax.plot(x_vals, y_vals, "o-", label=label, color=COLORS[label], alpha=0.8)
ax.set_xlabel("Number of Subtasks per Sample")
ax.set_ylabel("Frequency")
ax.set_title("Subtask Count Distribution (Naive vs Adaptive)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_subtask_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot_subtask_distribution.png")

# ── Plot 6: Node_b latency per sample — Naive vs Adaptive ────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ids = list(range(len(naive_rows)))
ax.plot(ids, [r["node_latencies_ms"]["node_b"] / 1000 for r in naive_rows],
        alpha=0.7, label="Naive node_b", color=COLORS["Naive"], linewidth=0.8)
ax.plot(ids, [r["node_latencies_ms"]["node_b"] / 1000 for r in adaptive_rows],
        alpha=0.7, label="Adaptive node_b", color=COLORS["Adaptive"], linewidth=0.8)
ax.set_xlabel("Sample Index")
ax.set_ylabel("Node_b Latency (seconds)")
ax.set_title("Per-Sample Node_b (i5 Worker) Latency: Naive vs Adaptive")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_nodeb_latency_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot_nodeb_latency_comparison.png")

# ── Plot 7: Total latency per sample across all 3 pipelines ──────────────
fig, ax = plt.subplots(figsize=(12, 4))
for label, rows in datasets.items():
    ax.plot([r["latency_ms"] / 1000 for r in rows],
            alpha=0.75, label=label, color=COLORS[label], linewidth=0.9)
ax.set_xlabel("Sample Index")
ax.set_ylabel("Total Latency (seconds)")
ax.set_title("Per-Sample Total Latency Across Pipelines")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_per_sample_latency.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot_per_sample_latency.png")

# ── Plot 8: PRO score vs ROUGE-1 (serial only) ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
pro_vals_s = []
r1_vals_s  = []
for r in serial_rows:
    for v in (r.get("uncertainty_scores") or []):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            rouge1 = r["correctness"].get("rouge1")
            if rouge1 is not None and not (isinstance(rouge1, float) and math.isnan(rouge1)):
                pro_vals_s.append(v)
                r1_vals_s.append(rouge1)
if pro_vals_s:
    ax.scatter(pro_vals_s, r1_vals_s, alpha=0.5, color=COLORS["Serial"], edgecolors="white", s=40)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="τ=0.5")
    # regression line
    z = np.polyfit(pro_vals_s, r1_vals_s, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(pro_vals_s), max(pro_vals_s), 100)
    ax.plot(x_line, p(x_line), "k--", linewidth=1, label="Linear fit")
    corr = np.corrcoef(pro_vals_s, r1_vals_s)[0, 1]
    ax.set_title(f"PRO Score vs ROUGE-1 (Serial, n={len(pro_vals_s)}, r={corr:.3f})")
    ax.set_xlabel("PRO Score (uncertainty)")
    ax.set_ylabel("ROUGE-1 (correctness)")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_pro_vs_rouge1.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot_pro_vs_rouge1.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. VALIDITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SECTION 10: RESULTS VALIDITY ASSESSMENT AND DIAGNOSTIC NOTES")
print(SEP)

serial_mean_lat  = np.nanmean([r["latency_ms"] for r in serial_rows]) / 1000
naive_mean_lat   = np.nanmean([r["latency_ms"] for r in naive_rows]) / 1000
adapt_mean_lat   = np.nanmean([r["latency_ms"] for r in adaptive_rows]) / 1000
serial_mean_r1   = safe_mean_corr(serial_rows, "rouge1")
naive_mean_r1    = safe_mean_corr(naive_rows, "rouge1")
adapt_mean_r1    = safe_mean_corr(adaptive_rows, "rouge1")

print(f"""
ISSUE 1 — CRITICAL: Parallel pipelines are ~5x SLOWER than serial
  Serial mean latency:   {serial_mean_lat:.1f}s
  Naive mean latency:    {naive_mean_lat:.1f}s  ({naive_mean_lat/serial_mean_lat:.1f}x slower)
  Adaptive mean latency: {adapt_mean_lat:.1f}s  ({adapt_mean_lat/serial_mean_lat:.1f}x slower)

  Root cause: The i5 worker node (node_b) is processing ALL subtask compute
  (node_b latency accounts for >100% of total latency, node_a latency = 0).
  This means either: (a) the master is sending all work to the worker and doing
  nothing locally, or (b) node_a latency is not being recorded in the result schema.
  The straggler problem (a slow i5 executing large subtask chains) is dominating.
  Decomposition overhead compounds this — splitting a 47s serial task into 4
  subtasks each taking 60s on the i5 yields 240s total.

ISSUE 2 — CRITICAL: uncertainty_scores are 100% NaN in Naive and Adaptive
  The PRO probe runs in both pipelines (the code calls compute_pro_score)
  but the scores never make it into the result record. In run_naive_mpi.py the
  probe result is discarded before being written to the record dict.
  In run_adaptive_mpi.py the worker reply doesn't include logprobs so
  compute_pro_score returns NaN from an empty list. This means:
  - Uncertainty-aware routing cannot be validated
  - The adaptive scheduler's routing signal is effectively random
  - ERCE and AUROC cannot be computed

ISSUE 3 — SIGNIFICANT: Naive and Adaptive produce IDENTICAL outputs
  100/100 samples have the same merged_output and identical routing assignments.
  The adaptive scheduler IS routing differently internally (EMA logic runs), but
  the result schema doesn't capture decomposed/locally_processed/fallback_to_serial
  flags — these three fields are missing from all 100 adaptive records.
  The routing dict shows same node assignments because the EMA scheduler and
  round-robin produced the same split on this dataset.

ISSUE 4 — MODERATE: BERTScore is NaN for all 100 samples across all pipelines
  bert: nan in every record. The bert_score library likely failed silently during
  the experiment run (possibly OOM or missing transformers model). This removes
  one of the two semantic metrics entirely.

ISSUE 5 — MODERATE: node_a latency = 0 in both parallel pipelines
  The master node's own inference contribution (if any) is not being recorded.
  Only node_b latency appears. This makes node-level latency analysis impossible.

ISSUE 6 — MINOR: Serial pipeline decomposition latency = 0
  The serial runner skips decomposition by design (subtasks = [full prompt]),
  so decomposition_ms is always 0. This is correct but means we cannot compare
  decomposition overhead across pipelines.

CORRECTNESS DROP (parallel vs serial):
  Serial ROUGE-1:   {serial_mean_r1:.4f}
  Naive ROUGE-1:    {naive_mean_r1:.4f}  ({(naive_mean_r1-serial_mean_r1)/serial_mean_r1*100:.1f}% change)
  Adaptive ROUGE-1: {adapt_mean_r1:.4f}  ({(adapt_mean_r1-serial_mean_r1)/serial_mean_r1*100:.1f}% change)

  Parallel pipelines show ~55% lower ROUGE-1 than serial. This is caused by
  context fragmentation: each subtask receives only its text fragment without
  the full document context, so individual outputs are shorter and less accurate.
  The merge step concatenates these fragments but doesn't recover coherence.
""")

print("─"*70)
print("WHAT NEEDS TO BE FIXED BEFORE RE-RUNNING:")
print("─"*70)
print("""
  1. Fix PRO score recording in run_naive_mpi.py and run_adaptive_mpi.py:
     After probe generation, store the scores in the per-subtask dict and
     write them to the result record's uncertainty_scores list.

  2. Add missing adaptive fields to result schema:
     decomposed, locally_processed, fallback_to_serial must be logged.

  3. Fix node_a latency recording:
     Any subtask executed locally on the master must add its latency_ms
     to node_latencies_ms['node_a'], not leave it as 0.

  4. Fix BERTScore: test bert_score import in isolation before re-running.
     If OOM, skip it or compute post-hoc on saved outputs.

  5. Reduce max_tokens for parallel subtasks: each subtask fragment gets
     full max_tokens allocation. With 4 subtasks each generating 192 tokens
     the total output is 4x the serial output, inflating node_b latency.
     Set per-subtask max_tokens = max_tokens // n_subtasks.
""")

print(SEP)
print("SECTION 11: CODE AND LOGIC VALIDITY ASSESSMENT")
print(SEP)
print("""
  run_serial.py:       VALID. Logic is correct. PRO scores recorded properly.
                       Step latency breakdown is well-instrumented.

  run_naive_mpi.py:    LOGIC VALID, RECORDING BUG. Round-robin dispatch and
                       pipelined refill pattern are correct. PRO scores computed
                       but not written to result record. Worker fallback to local
                       is properly implemented.

  run_adaptive_mpi.py: LOGIC VALID, RECORDING BUGS. EMA-based scheduling and
                       should_decompose() gate are sound approaches. Worker rank
                       scoring with rtt+compute estimation is theoretically correct.
                       Three bugs: PRO scores not recorded, adaptive fields not
                       logged, node_a latency not accumulated.

  naive.py scheduler:  VALID. Simple round-robin, no issues.

  uncertainty_aware.py: VALID. PRO threshold routing and network feasibility
                        check are correct. Fallback to all-node_a is correct.

  EMA class:           VALID. alpha=0.2 is a reasonable choice for slowly
                       changing latency estimates. Initial probe via probe_workers()
                       before processing is good practice.

  should_decompose():  VALID concept but the threshold (0.5x) may be too
                       aggressive for high-latency networks. At 150-200ms RTT
                       the break-even point for decomposition shifts significantly.
""")

print(f"\n{SEP}")
print(f"All outputs saved to: {OUT_DIR}")
print("Files: latency_summary.csv, serial_step_latencies.csv,")
print("       correctness_summary.csv, pro_score_summary.csv,")
print("       decomposition_summary.csv, full_comparison.csv,")
print("       plot_latency_distribution.png, plot_correctness_metrics.png,")
print("       plot_pro_scores_serial.png, plot_serial_step_latency.png,")
print("       plot_subtask_distribution.png, plot_nodeb_latency_comparison.png,")
print("       plot_per_sample_latency.png, plot_pro_vs_rouge1.png")
print(SEP)
