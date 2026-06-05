"""Dataset-wide report figures for the phase 1 writeup.

Seven figures, each tied to a specific §8 hypothesis claim. All read from
already-saved artifacts (per-seq JSON dumps, saved .npy predictions,
boundary-threshold CSV, correlations cache) — no inference required.

Color / label convention used everywhere:
    RAFT-32        → blue   (#3F6FD9)
    GMFlow-basic   → orange (#E07A2C)
    GMFlow-refine  → green  (#3FA666)

Outputs under results/figures/report/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from cvflow.datasets.middlebury import Middlebury
from cvflow.metrics.middlebury import angular_error_deg, gt_valid_mask


# ============================================================================
#  conventions
# ============================================================================
LABELS = {
    "raft":          "RAFT-32",
    "gmflow_basic":  "GMFlow-basic",
    "gmflow_refine": "GMFlow-refine",
}
COLORS = {
    "raft":          "#3F6FD9",
    "gmflow_basic":  "#E07A2C",
    "gmflow_refine": "#3FA666",
}
TAGS = {
    "raft":          "raft-raft-things-iter32",
    "gmflow_basic":  "gmflow-gmflow_things-e9887eda",
    "gmflow_refine": "gmflow-gmflow_with_refine_things-36579974",
}
REGIONS = ["all", "matched", "unmatched", "s0_1", "s0_10", "s10_40", "s40+", "s60+",
           "disc", "untex", "blur"]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":    "white",
        "axes.facecolor":      "#fbfbfb",
        "axes.edgecolor":      "#555555",
        "axes.linewidth":      0.9,
        "axes.titlesize":      13,
        "axes.titleweight":    "bold",
        "axes.titlepad":       12,
        "axes.labelsize":      11,
        "axes.labelpad":       6,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "axes.axisbelow":      True,
        "grid.color":          "#dddddd",
        "grid.linewidth":      0.6,
        "grid.linestyle":      "-",
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "xtick.color":         "#444444",
        "ytick.color":         "#444444",
        "legend.frameon":      True,
        "legend.facecolor":    "white",
        "legend.edgecolor":    "#bbbbbb",
        "legend.framealpha":   0.95,
        "legend.fontsize":     10,
        "legend.title_fontsize": 10,
        "font.family":         "DejaVu Sans",
        "savefig.dpi":         150,
        "savefig.bbox":        "tight",
        "savefig.facecolor":   "white",
        "figure.dpi":          110,
    })


def load_per_seq(name: str, pass_: str, root: Path) -> dict:
    return json.loads((root / "per_seq_stats" / f"{name}_{pass_}.json").read_text())


def weighted_mean_epe(stats: dict, mask: str) -> float:
    num = sum(stats[mask][s]["sum_epe"] for s in stats[mask])
    den = sum(stats[mask][s]["sum_n"] for s in stats[mask])
    return num / den if den > 0 else float("nan")


def annotate_bars(ax, bars, values, fmt: str = "{:.2f}", offset: float = 1.02,
                  fontsize: int = 8, color: str = "#222") -> None:
    """Label each bar with its value."""
    for rect, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        y = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, y * offset, fmt.format(v),
                ha="center", va="bottom", fontsize=fontsize, color=color)


# ============================================================================
#  figure 1 — Pareto: latency vs EPE
# ============================================================================
def fig_pareto(out: Path) -> None:
    iter_sweep = [
        (4,  198, 1.910),
        (8,  276, 1.593),
        (12, 358, 1.510),
        (32, 765, 1.446),
    ]
    n50 = [
        ("raft",          594, 1.446),
        ("gmflow_basic",  252, 1.484),
        ("gmflow_refine", 652, 1.073),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    xs = [r[1] for r in iter_sweep]
    ys = [r[2] for r in iter_sweep]
    ax.plot(xs, ys, color=COLORS["raft"], alpha=0.55, linestyle="--",
            marker="o", markersize=9, markerfacecolor="white",
            markeredgewidth=1.8, linewidth=1.4, zorder=2,
            label="RAFT iter sweep (4/8/12/32 iters — iter-sweep batch)")
    for it, x, y in iter_sweep:
        ax.annotate(f"  {it} it.", (x, y), fontsize=9.5, va="center",
                    color=COLORS["raft"], alpha=0.9)
    for key, x, y in n50:
        ax.scatter([x], [y], s=270, color=COLORS[key], edgecolor="white",
                   linewidth=2.5, zorder=5, marker="D")
        ax.scatter([x], [y], s=270, facecolor="none",
                   edgecolor="#222", linewidth=1.0, zorder=6, marker="D")
        ax.annotate(f"  {LABELS[key]}", (x, y), fontsize=11,
                    va="center", fontweight="bold")

    from matplotlib.lines import Line2D
    leg_elems = [
        Line2D([0], [0], color=COLORS["raft"], alpha=0.55, linestyle="--",
               marker="o", markersize=9, markerfacecolor="white",
               markeredgewidth=1.8, label="RAFT iter sweep (iter-sweep batch)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["raft"],
               markeredgecolor="white", markeredgewidth=2, markersize=14, label=LABELS["raft"]),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["gmflow_basic"],
               markeredgecolor="white", markeredgewidth=2, markersize=14, label=LABELS["gmflow_basic"]),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["gmflow_refine"],
               markeredgecolor="white", markeredgewidth=2, markersize=14, label=LABELS["gmflow_refine"]),
    ]
    ax.legend(handles=leg_elems, loc="upper right", title="n=50 batch (cross-model)")
    ax.set_xlabel("Per-pair latency (ms) — 1024×436, RTX 3050 Ti Laptop")
    ax.set_ylabel("Sintel-clean EPE / all (px)")
    ax.set_title("Latency vs Accuracy on Sintel clean (full 1041 pairs)")
    ax.set_xlim(left=130)
    txt = ("Pareto frontier (n=50 batch):  GMFlow-basic  →  RAFT-32  →  GMFlow-refine\n"
           "•  GMFlow-basic strictly dominates RAFT-12 (faster + more accurate)\n"
           "•  Refine 10% slower than RAFT-32 but 26% more accurate")
    ax.text(0.02, 0.04, txt, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#bbb", alpha=0.95))
    fig.tight_layout()
    fig.savefig(out / "pareto.png")
    plt.close(fig)
    print(f"  wrote {out / 'pareto.png'}")


# ============================================================================
#  figure 2 — Region bars (full dataset)
# ============================================================================
def fig_region_bars(out: Path, root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=False)
    for ax, pass_ in zip(axes, ("clean", "final")):
        stats = {k: load_per_seq(k, pass_, root) for k in LABELS}
        x = np.arange(len(REGIONS))
        width = 0.27
        for i, key in enumerate(("raft", "gmflow_basic", "gmflow_refine")):
            vals = [weighted_mean_epe(stats[key], r) for r in REGIONS]
            ax.bar(x + (i - 1) * width, vals, width,
                   color=COLORS[key], edgecolor="white", linewidth=0.6,
                   label=LABELS[key], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(REGIONS, rotation=35, ha="right")
        ax.set_ylabel("Mean EPE (px)" if pass_ == "clean" else "")
        ax.set_title(f"Sintel {pass_} — full dataset (1041 pairs)")
        ax.set_yscale("log")
        ax.legend(loc="upper left")
    fig.suptitle("Per-region EPE — lower is better  ·  log y-axis  ·  "
                 "11 masks × 3 models × 2 passes",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "region_bars_full.png")
    plt.close(fig)
    print(f"  wrote {out / 'region_bars_full.png'}")


# ============================================================================
#  figure 3 — Boundary F1 threshold sensitivity
# ============================================================================
def fig_boundary_f1(out: Path, root: Path) -> None:
    csv_path = root / "figures" / "boundary_threshold" / "sensitivity.csv"
    rows = list(csv.DictReader(csv_path.open()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, pass_ in zip(axes, ("clean", "final")):
        for key in ("raft", "gmflow_basic", "gmflow_refine"):
            tag = TAGS[key]
            taus, f1s = [], []
            for r in rows:
                if r["tag"] == tag and r["pass"] == pass_:
                    taus.append(float(r["threshold"]))
                    f1s.append(float(r["F1"]))
            ts, fs = zip(*sorted(zip(taus, f1s)))
            ax.plot(ts, fs, marker="o", markersize=11, linewidth=2.4,
                    markerfacecolor="white", markeredgewidth=2.2,
                    color=COLORS[key], label=LABELS[key], zorder=3)
            for t, f in zip(ts, fs):
                ax.annotate(f"{f:.3f}", (t, f), xytext=(0, 14), textcoords="offset points",
                            ha="center", fontsize=9, color=COLORS[key], fontweight="bold")
        ax.set_xticks([0.5, 1.0, 2.0])
        ax.set_xlabel("Gradient-magnitude threshold τ for boundary mask")
        if pass_ == "clean":
            ax.set_ylabel("Mean boundary F1  (higher is better)")
        ax.set_title(f"Sintel {pass_}")
        ax.legend(loc="lower left")
    fig.suptitle("Boundary-F1 threshold sensitivity (H3)  ·  "
                 "refine > RAFT > basic at every τ",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "boundary_f1_sensitivity.png")
    plt.close(fig)
    print(f"  wrote {out / 'boundary_f1_sensitivity.png'}")


# ============================================================================
#  figure 4 — Middlebury per-sequence (3 models)
# ============================================================================
def _compute_middlebury_stats(middlebury_root: Path):
    """Return (per_seq_epe, per_seq_pearson, per_seq_spearman) for each model."""
    ds = Middlebury(middlebury_root)
    epe_by = {k: {} for k in TAGS}
    pearson_by = {k: {} for k in TAGS}
    spearman_by = {k: {} for k in TAGS}
    for key, tag in TAGS.items():
        for p in ds.pairs():
            pred = np.load(Path("results/predictions") / tag / "middlebury" / p.seq / "flow10.npy")
            valid = gt_valid_mask(p.gt_flow)
            e = np.linalg.norm(pred - p.gt_flow, axis=-1)[valid].astype(np.float64)
            a = angular_error_deg(pred, p.gt_flow)[valid].astype(np.float64)
            epe_by[key][p.seq] = float(e.mean())
            pearson_by[key][p.seq]  = float(np.corrcoef(e, a)[0, 1]) if e.std() > 0 and a.std() > 0 else float("nan")
            spearman_by[key][p.seq] = float(spearmanr(e, a).statistic)
    return epe_by, pearson_by, spearman_by


def fig_middlebury(out: Path, middlebury_root: Path,
                   epe_by, _, __) -> None:
    seqs = sorted(epe_by["raft"].keys())
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(seqs) + 1)
    width = 0.27
    for i, key in enumerate(("raft", "gmflow_basic", "gmflow_refine")):
        vals = [epe_by[key][s] for s in seqs]
        vals.append(float(np.mean(vals)))
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      color=COLORS[key], edgecolor="white", linewidth=0.6,
                      label=LABELS[key], zorder=3)
        annotate_bars(ax, bars, vals, fmt="{:.2f}", offset=1.01, fontsize=7.5)
    ax.set_xticks(x)
    labels = seqs + ["MEAN"]
    ax.set_xticklabels(labels, rotation=25, ha="right")
    for tick, lab in zip(ax.get_xticklabels(), labels):
        if lab == "MEAN":
            tick.set_fontweight("bold")
    ax.set_ylabel("Mean EPE (px) on valid pixels")
    ax.set_title("Middlebury 'other' — zero-shot Things-trained checkpoints (H10)")
    ax.legend(loc="upper left")
    fig.text(0.5, -0.04,
             "RAFT wins all 8 sequences + the MEAN.  Refine beats basic on 7/8 (Venus is the outlier).",
             ha="center", fontsize=10, color="#333")
    fig.tight_layout()
    fig.savefig(out / "middlebury_per_seq.png")
    plt.close(fig)
    print(f"  wrote {out / 'middlebury_per_seq.png'}")


# ============================================================================
#  figure 5 — Clean → Final paired Δ distribution
# ============================================================================
def fig_clean_to_final(out: Path, root: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    width = 0.27
    keys = ("raft", "gmflow_basic", "gmflow_refine")
    deltas_by_key: dict[str, list[float]] = {}
    seqs = None
    for key in keys:
        clean = load_per_seq(key, "clean", root)["all"]
        final = load_per_seq(key, "final", root)["all"]
        if seqs is None:
            seqs = sorted(set(clean) & set(final))
        deltas_by_key[key] = [
            (final[s]["sum_epe"] / final[s]["sum_n"]) - (clean[s]["sum_epe"] / clean[s]["sum_n"])
            for s in seqs
        ]
    x = np.arange(len(seqs))
    for i, key in enumerate(keys):
        vals = deltas_by_key[key]
        ax.bar(x + (i - 1) * width, vals, width,
               color=COLORS[key], edgecolor="white", linewidth=0.6,
               label=f"{LABELS[key]}   mean per-seq Δ = {np.mean(vals):+.2f}", zorder=3)
    ax.axhline(0, color="#222", linewidth=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(seqs, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Per-sequence ΔEPE  =  EPE_final − EPE_clean (px)")
    ax.set_title("Clean → Final degradation per sequence (H7)")
    fig.text(0.5, -0.04,
             "Negative bars = sequence got EASIER on Final (motion blur smooths errors).  "
             "ambush_2 dominates the degradation budget.",
             ha="center", fontsize=10, color="#333")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "clean_to_final_delta.png")
    plt.close(fig)
    print(f"  wrote {out / 'clean_to_final_delta.png'}")


# ============================================================================
#  figure 6 — AE↔EPE correlation per mask (Sintel)
# ============================================================================
def fig_correlation_per_mask(out: Path, corr_path: Path) -> None:
    data = json.loads(corr_path.read_text())
    # One big tall figure: 4 panels stacked 2x2, generous size for 11 region labels each.
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharey=True)
    for col, pass_ in enumerate(("clean", "final")):
        for row, stat in enumerate(("pearson", "spearman")):
            ax = axes[row, col]
            x = np.arange(len(REGIONS))
            width = 0.27
            for i, key in enumerate(("raft", "gmflow_basic", "gmflow_refine")):
                vals = [data[f"{key}_{pass_}"][m][stat] for m in REGIONS]
                ax.bar(x + (i - 1) * width, vals, width,
                       color=COLORS[key], edgecolor="white", linewidth=0.6,
                       label=LABELS[key], zorder=3)
            ax.axhline(0, color="#222", linewidth=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(REGIONS, rotation=35, ha="right", fontsize=11)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{stat.capitalize()}(EPE, AE) · Sintel {pass_}", fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{stat.capitalize()} correlation", fontsize=12)
            if row == 0 and col == 0:
                ax.legend(loc="lower left", ncol=1, fontsize=11)
    fig.suptitle("AE ↔ EPE correlation per region — Pearson (top) vs Spearman (bottom)",
                 fontsize=16, fontweight="bold", y=1.00)
    fig.text(0.5, -0.015,
             "High = AE rankings track EPE rankings tightly within that mask.  "
             "Low = AE and EPE disagree on which pixels are 'worst'.",
             ha="center", fontsize=11, color="#333")
    fig.tight_layout()
    fig.savefig(out / "ae_epe_correlation_per_mask.png")
    plt.close(fig)
    print(f"  wrote {out / 'ae_epe_correlation_per_mask.png'}")


# ============================================================================
#  figure 7 — AE↔EPE correlation per Middlebury sequence
# ============================================================================
def fig_correlation_middlebury(out: Path, pearson_by, spearman_by) -> None:
    seqs = sorted(pearson_by["raft"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, by, name in [(axes[0], pearson_by, "Pearson"),
                          (axes[1], spearman_by, "Spearman")]:
        x = np.arange(len(seqs))
        width = 0.27
        for i, key in enumerate(("raft", "gmflow_basic", "gmflow_refine")):
            vals = [by[key][s] for s in seqs]
            ax.bar(x + (i - 1) * width, vals, width,
                   color=COLORS[key], edgecolor="white", linewidth=0.6,
                   label=LABELS[key], zorder=3)
        ax.axhline(0, color="#222", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(seqs, rotation=30, ha="right")
        ax.set_title(f"{name} (EPE, AE) per sequence")
        if name == "Pearson":
            ax.set_ylabel("correlation coefficient")
        ax.legend(loc="lower left")
    fig.suptitle("Middlebury — AE ↔ EPE correlation per sequence",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
             "Urban2 is the weakest correlation cell — small-flow regime where AE noise "
             "dominates EPE signal.",
             ha="center", fontsize=10, color="#333")
    fig.tight_layout()
    fig.savefig(out / "ae_epe_correlation_middlebury.png")
    plt.close(fig)
    print(f"  wrote {out / 'ae_epe_correlation_middlebury.png'}")


# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seq-root", default="results")
    ap.add_argument("--middlebury-root", default="datasets/Middleburry")
    ap.add_argument("--out", default="results/figures/report")
    ap.add_argument("--corr-cache", default="results/correlations/sintel_per_mask.json")
    args = ap.parse_args()

    setup_style()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.per_seq_root)
    middlebury_root = Path(args.middlebury_root)

    fig_pareto(out)
    fig_region_bars(out, root)
    fig_boundary_f1(out, root)
    epe_by, pearson_by, spearman_by = _compute_middlebury_stats(middlebury_root)
    fig_middlebury(out, middlebury_root, epe_by, pearson_by, spearman_by)
    fig_clean_to_final(out, root)
    if Path(args.corr_cache).exists():
        fig_correlation_per_mask(out, Path(args.corr_cache))
    else:
        print(f"  SKIP correlation_per_mask — cache missing ({args.corr_cache}). "
              f"Run: python -m cvflow.runners.dump_correlations")
    fig_correlation_middlebury(out, pearson_by, spearman_by)

    print("\nAll figures written to", out)


if __name__ == "__main__":
    main()
