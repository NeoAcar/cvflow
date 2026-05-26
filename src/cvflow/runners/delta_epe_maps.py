"""Per-pixel ΔEPE = EPE_final − EPE_clean, aggregated per sequence.

Reads saved .npy predictions on both passes, emits:
  - results/figures/delta_epe/<model>/<seq>.png  — mean ΔEPE map (colormap)
  - results/figures/delta_epe/<model>/_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvflow.datasets.sintel import Sintel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--out", default="results/figures/delta_epe")
    args = ap.parse_args()

    tag = Path(args.pred_root).name
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = Sintel(args.sintel_root, pass_="clean")
    final = Sintel(args.sintel_root, pass_="final")
    pred_clean_root = Path(args.pred_root) / "sintel" / "clean"
    pred_final_root = Path(args.pred_root) / "sintel" / "final"

    rows = []
    for seq in clean.seqs:
        accum = None
        n = 0
        for pc, pf in zip(clean.pairs(seqs=[seq]), final.pairs(seqs=[seq])):
            pred_c = np.load(pred_clean_root / seq / f"frame_{pc.idx:04d}.npy")
            pred_f = np.load(pred_final_root / seq / f"frame_{pf.idx:04d}.npy")
            valid = pc.invalid == 0
            ee_c = np.linalg.norm(pred_c - pc.gt_flow, axis=-1)
            ee_f = np.linalg.norm(pred_f - pf.gt_flow, axis=-1)
            delta = (ee_f - ee_c) * valid.astype(np.float32)
            if accum is None:
                accum = np.zeros_like(delta)
            accum += delta
            n += 1
        mean_delta = accum / n

        d_pos = mean_delta[mean_delta > 0]
        d_neg = mean_delta[mean_delta < 0]
        rows.append({
            "seq":      seq,
            "n_pairs":  n,
            "mean":     float(mean_delta.mean()),
            "median":   float(np.median(mean_delta)),
            "p95_abs":  float(np.percentile(np.abs(mean_delta), 95)),
            "frac_worse_on_final": float((mean_delta > 0).mean()),
            "mean_pos": float(d_pos.mean()) if d_pos.size else 0.0,
            "mean_neg": float(d_neg.mean()) if d_neg.size else 0.0,
        })

        # diverging colormap centered at 0
        absmax = float(np.percentile(np.abs(mean_delta), 99))
        absmax = max(absmax, 1e-3)
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(mean_delta, cmap="RdBu_r", vmin=-absmax, vmax=absmax)
        ax.set_title(f"{tag}  {seq}  ΔEPE = EPE_final − EPE_clean   (n={n} pairs)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ΔEPE (px)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{seq}.png", dpi=110)
        plt.close(fig)
        print(f"  {seq:14s} mean Δ={rows[-1]['mean']:+.3f}  median={rows[-1]['median']:+.3f}  "
              f"frac_worse={rows[-1]['frac_worse_on_final']:.2%}")

    csv_path = out_dir / "_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")
    print(f"wrote {len(rows)} PNGs to {out_dir}/")


if __name__ == "__main__":
    main()
