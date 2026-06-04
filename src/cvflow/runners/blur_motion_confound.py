"""Test whether the blur mask is a hidden motion-magnitude proxy.

For each Sintel-clean sequence: compute (a) blur-fraction = mean over all valid
pixels of `blur_mask(img1) & invalid==0`, (b) mean GT-flow magnitude. Pearson
correlation across the 23 sequences quantifies the confound.

If r is large (≥ 0.7) and the linear fit is steep, the blur mask is acting as a
hidden motion-magnitude detector, and §7d's "GMFlow handles blur better" reading
is contaminated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from cvflow.datasets.sintel import Sintel
from cvflow.masks.blur import blur_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--pass", dest="pass_", choices=["clean", "final"], default="clean")
    ap.add_argument("--blur-window", type=int, default=7)
    ap.add_argument("--blur-thresh", type=float, default=20.0)
    ap.add_argument("--out", default="results/figures/blur_motion_confound.png")
    args = ap.parse_args()

    ds = Sintel(args.sintel_root, pass_=args.pass_)
    by_seq: dict[str, dict[str, float]] = {}
    for p in ds.pairs():
        b = blur_mask(p.img1, window=args.blur_window, var_thresh=args.blur_thresh)
        valid = p.invalid == 0
        if not valid.any():
            continue
        speed = np.linalg.norm(p.gt_flow, axis=-1)[valid]
        blur_frac = float((b & valid).sum()) / float(valid.sum())
        d = by_seq.setdefault(p.seq, {"sum_blur": 0.0, "sum_speed": 0.0, "n": 0})
        d["sum_blur"] += blur_frac
        d["sum_speed"] += float(speed.mean())
        d["n"] += 1

    seqs = sorted(by_seq.keys())
    blur_fracs = np.array([by_seq[s]["sum_blur"] / by_seq[s]["n"] for s in seqs])
    mean_speeds = np.array([by_seq[s]["sum_speed"] / by_seq[s]["n"] for s in seqs])
    pearson = float(np.corrcoef(blur_fracs, mean_speeds)[0, 1])
    spearman = float(spearmanr(blur_fracs, mean_speeds).statistic)

    print(f"\nBlur ↔ mean GT-flow correlation across {len(seqs)} sequences (pass={args.pass_}):")
    print(f"  Pearson  = {pearson:.4f}")
    print(f"  Spearman = {spearman:.4f}\n")
    print(f"  {'sequence':<14s}  {'blur_frac':>10s}  {'mean_speed':>10s}")
    for i, s in enumerate(seqs):
        print(f"  {s:<14s}  {blur_fracs[i]:>10.4f}  {mean_speeds[i]:>10.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_speeds, blur_fracs, c="C0", s=40, edgecolor="black", linewidth=0.5)
    for i, s in enumerate(seqs):
        ax.annotate(s, (mean_speeds[i], blur_fracs[i]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("per-sequence mean |gt_flow| (px)")
    ax.set_ylabel("per-sequence blur-fraction")
    ax.set_title(f"Blur mask vs motion magnitude ({args.pass_})\n"
                 f"Pearson = {pearson:.3f}, Spearman = {spearman:.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
