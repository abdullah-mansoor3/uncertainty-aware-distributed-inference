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

# Step latencies for all pipelines
print("\n\nStep Latency Breakdown by Pipeline:")
step_rows = []
for label, rows in datasets.items():
    steps_agg = {}
    for r in rows:
        for k, v in (r.get("step_latencies_ms") or {}).items():
            try:
                steps_agg.setdefault(k, []).append(float(v))
            except Exception:
                continue
    for step_name, values in steps_agg.items():
        if not values:
            continue
        step_rows.append(
            {
                "Pipeline": label,
                "Step": step_name,
                "Mean (ms)": int(np.mean(values)),
                "P95 (ms)": int(np.percentile(values, 95)),
                "Max (ms)": int(np.max(values)),
            }
        )
step_df = pd.DataFrame(step_rows)
if not step_df.empty:
    print(step_df.sort_values(["Pipeline", "Mean (ms)"], ascending=[True, False]).to_string(index=False))
else:
    print("No step_latencies_ms field found in results.")

# Save
lat_df.to_csv(OUT_DIR / "latency_summary.csv", index=False)
step_df.to_csv(OUT_DIR / "step_latency_summary.csv", index=False)
print(f"\nSaved: latency_summary.csv, step_latency_summary.csv")

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
for _, row in pro_df.iterrows():
    if int(row["Valid"]) == 0 and int(row["Total Scores"]) > 0:
        print(f"  NOTE: {row['Pipeline']} has zero valid PRO scores despite populated uncertainty_scores.")

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

# Decomposition alignment quality (requires decomposition_ground_truth-derived score in results)
decomp_align_rows = []
for label, rows in datasets.items():
    vals = [
        float(r.get("decomposition_alignment_score"))
        for r in rows
        if r.get("decomposition_alignment_score") is not None
        and not (isinstance(r.get("decomposition_alignment_score"), float) and math.isnan(r.get("decomposition_alignment_score")))
    ]
    decomp_align_rows.append(
        {
            "Pipeline": label,
            "N valid": len(vals),
            "Mean": round(float(np.mean(vals)), 4) if vals else float("nan"),
            "P50": round(float(np.percentile(vals, 50)), 4) if vals else float("nan"),
            "P95": round(float(np.percentile(vals, 95)), 4) if vals else float("nan"),
        }
    )
decomp_align_df = pd.DataFrame(decomp_align_rows)
print("\nDecomposition Alignment Score Summary:")
print(decomp_align_df.to_string(index=False))
decomp_align_df.to_csv(OUT_DIR / "decomposition_alignment_summary.csv", index=False)
print("Saved: decomposition_alignment_summary.csv")

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

# Calibration metrics tracked during MPI runs
def latest_running_metric(rows, key):
    vals = [
        float(r.get(key))
        for r in rows
        if r.get(key) is not None and not (isinstance(r.get(key), float) and math.isnan(r.get(key)))
    ]
    return vals[-1] if vals else float("nan")

calib_df = pd.DataFrame(
    [
        {"Pipeline": "Serial", "ERCE": float("nan"), "AUROC": float("nan")},
        {"Pipeline": "Naive", "ERCE": latest_running_metric(naive_rows, "running_erce"), "AUROC": latest_running_metric(naive_rows, "running_auroc")},
        {"Pipeline": "Adaptive", "ERCE": latest_running_metric(adaptive_rows, "running_erce"), "AUROC": latest_running_metric(adaptive_rows, "running_auroc")},
    ]
)
print("\nCalibration Summary (latest running values):")
print(calib_df.to_string(index=False))
calib_df.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
print("Saved: calibration_summary.csv")

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
        "ERCE":              round(latest_running_metric(rows, "running_erce"), 4) if label != "Serial" else float("nan"),
        "AUROC":             round(latest_running_metric(rows, "running_auroc"), 4) if label != "Serial" else float("nan"),
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
    ax.set_title("PRO Score Distribution — Serial Pipeline")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_pro_scores_serial.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot_pro_scores_serial.png")

# ── Plot 4: Serial step latency breakdown ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
steps_agg = {}
for r in serial_rows:
    for k, v in (r.get("step_latencies_ms") or {}).items():
        try:
            steps_agg.setdefault(k, []).append(float(v))
        except Exception:
            continue
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
LATENCY OVERVIEW
  Serial mean latency:   {serial_mean_lat:.1f}s
  Naive mean latency:    {naive_mean_lat:.1f}s  ({naive_mean_lat/serial_mean_lat:.2f}x vs serial)
  Adaptive mean latency: {adapt_mean_lat:.1f}s  ({adapt_mean_lat/serial_mean_lat:.2f}x vs serial)

CORRECTNESS OVERVIEW (ROUGE-1)
  Serial:   {serial_mean_r1:.4f}
  Naive:    {naive_mean_r1:.4f}  ({(naive_mean_r1-serial_mean_r1)/serial_mean_r1*100:.1f}% vs serial)
  Adaptive: {adapt_mean_r1:.4f}  ({(adapt_mean_r1-serial_mean_r1)/serial_mean_r1*100:.1f}% vs serial)
""")

print("─"*70)
print("AUTOMATED DATA QUALITY CHECKS:")
print("─"*70)
for label, rows in datasets.items():
    bert_nan = sum(
        1
        for r in rows
        if isinstance(r.get("correctness", {}).get("bert"), float) and math.isnan(r.get("correctness", {}).get("bert"))
    )
    pro_vals = [v for r in rows for v in (r.get("uncertainty_scores") or []) if v is not None]
    pro_valid = [v for v in pro_vals if not (isinstance(v, float) and math.isnan(v))]
    has_step_latency = sum(1 for r in rows if isinstance(r.get("step_latencies_ms"), dict))
    print(
        f"  {label:<8} | BERT NaN: {bert_nan}/{len(rows)}"
        f" | Valid PRO: {len(pro_valid)}/{len(pro_vals)}"
        f" | step_latencies_ms present: {has_step_latency}/{len(rows)}"
    )

print(SEP)
print("SECTION 11: UPDATED DIAGNOSTIC SUMMARY")
print(SEP)
if not step_df.empty:
    print("Top latency contributors by pipeline (highest mean step first):")
    top_steps = step_df.sort_values(["Pipeline", "Mean (ms)"], ascending=[True, False]).groupby("Pipeline").head(3)
    print(top_steps.to_string(index=False))
else:
    print("No step-latency data found. Re-run experiments with step_latencies_ms logging enabled.")

print(f"\n{SEP}")
print(f"All outputs saved to: {OUT_DIR}")
print("Files: latency_summary.csv, step_latency_summary.csv,")
print("       correctness_summary.csv, pro_score_summary.csv, calibration_summary.csv,")
print("       decomposition_summary.csv, decomposition_alignment_summary.csv, full_comparison.csv,")
print("       plot_latency_distribution.png, plot_correctness_metrics.png,")
print("       plot_pro_scores_serial.png, plot_serial_step_latency.png,")
print("       plot_subtask_distribution.png, plot_nodeb_latency_comparison.png,")
print("       plot_per_sample_latency.png, plot_pro_vs_rouge1.png")
print(SEP)
