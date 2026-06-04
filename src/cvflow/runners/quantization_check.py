"""Test the §3b 'GMFlow predicts on a 1/8-pixel grid' claim.

If GMFlow's convex upsample leaves a quantization residual at 1/8 px, the
histogram of matched-pixel EPE on slow sequences should show spikes near
k/8 multiples (0, 0.125, 0.25, 0.375, ...). On fast sequences the artifact
should be smeared by the larger motion errors.

For each (tag, sequence) we plot a histogram of matched-pixel EPE on a
single representative frame pair (frame_0001) and compute the fraction of
matched pixels falling within ±0.02 of each k/8 multiple as a quick
quantitative summary. Output: PNG + small summary CSV.
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
    ap.add_argument("--tags", nargs="+", default=[
        "gmflow-gmflow_things-e9887eda",
        "gmflow-gmflow_with_refine_things-36579974",
        "raft-raft-things-iter32",
    ])
    ap.add_argument("--slow-seqs", nargs="+", default=["alley_2", "bandage_2"],
                    help="Slow-motion sequences expected to expose 1/8-grid residual")
    ap.add_argument("--fast-seq", default="ambush_4",
                    help="Fast-motion control where the artifact should be smeared")
    ap.add_argument("--frame", type=int, default=1, help="Frame index (1-based)")
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--max-epe", type=float, default=1.0)
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--out", default="results/figures/quantization")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = Sintel(args.sintel_root, pass_="clean")
    all_seqs = list(args.slow_seqs) + [args.fast_seq]
    pair_for_seq = {}
    for seq in all_seqs:
        for p in ds.pairs(seqs=[seq]):
            if p.idx == args.frame:
                pair_for_seq[seq] = p
                break

    rows: list[dict] = []
    targets_k8 = [k / 8.0 for k in range(0, 9)]  # 0, 0.125, ..., 1.0

    for tag in args.tags:
        for seq in all_seqs:
            pair = pair_for_seq[seq]
            pred_path = Path("results/predictions") / tag / "sintel" / "clean" / seq / f"frame_{args.frame:04d}.npy"
            if not pred_path.exists():
                print(f"  skip {tag} / {seq} — {pred_path} missing")
                continue
            pred = np.load(pred_path)
            valid = (pair.invalid == 0) & (pair.occlusion == 0)  # matched
            epe = np.linalg.norm(pred - pair.gt_flow, axis=-1)[valid]
            epe_clipped = epe[epe <= args.max_epe]

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(epe_clipped, bins=args.bins, range=(0, args.max_epe),
                    color="C0", edgecolor="none", alpha=0.85)
            for k in range(1, 9):
                ax.axvline(k / 8.0, color="red", alpha=0.35, linestyle="--", linewidth=0.8)
            kind = "slow" if seq in args.slow_seqs else "fast"
            ax.set_title(f"{tag} | {seq} (frame {args.frame}, matched pixels) — {kind}\n"
                         f"red dashed: k/8 px (k=1..8)")
            ax.set_xlabel("matched-pixel EPE (px)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fname = out / f"{tag}__{seq}__frame{args.frame:04d}.png"
            fig.savefig(fname, dpi=120)
            plt.close(fig)

            # Pile-up summary: fraction within ±0.02 of each k/8 (k=0..8)
            tol = 0.02
            buckets = {f"frac_near_{k}/8": float(np.mean(np.abs(epe - t) <= tol))
                       for k, t in enumerate(targets_k8)}
            row = {"tag": tag, "seq": seq, "kind": kind, "n_matched": int(epe.size),
                   "median_epe": float(np.median(epe)),
                   "mean_epe":   float(epe.mean())}
            row.update(buckets)
            rows.append(row)
            print(f"  wrote {fname.name}")

    csv_path = out / "summary.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {csv_path}")
    else:
        print("\nno rows — check predictions exist")


if __name__ == "__main__":
    main()
