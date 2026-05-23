"""Offline evaluation: load saved .npy predictions + Sintel GT, compute the full
mask suite (All/Matched/Unmatched/s0_10/s10_40/s40+/Disc/Untex) + boundary F-score.

No GPU required.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cvflow.datasets.sintel import Sintel
from cvflow.masks.motion_boundary import disc_mask
from cvflow.masks.textureless import untext_mask
from cvflow.metrics.boundary_fscore import boundary_fscore
from cvflow.metrics.sintel import SintelMetrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True, help="e.g. results/predictions/raft-raft-things-iter32")
    ap.add_argument("--pass", dest="pass_", choices=["clean", "final"], default="clean")
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--disc-thresh", type=float, default=1.0)
    ap.add_argument("--untex-thresh", type=float, default=5.0)
    args = ap.parse_args()

    pred_dir = Path(args.pred_root) / "sintel" / args.pass_
    ds = Sintel(args.sintel_root, pass_=args.pass_)
    m = SintelMetrics()
    fscore_per_seq: dict[str, list[float]] = defaultdict(list)
    f_sum, p_sum, r_sum, n = 0.0, 0.0, 0.0, 0

    t0 = time.time()
    for i, pair in enumerate(ds.pairs()):
        npy = pred_dir / pair.seq / f"frame_{pair.idx:04d}.npy"
        pred = np.load(npy)
        disc = disc_mask(pair.gt_flow, args.disc_thresh)
        untex = untext_mask(pair.img1, args.untex_thresh)
        m.update(pred, pair.gt_flow, pair.occlusion, pair.invalid, pair.seq, disc=disc, untex=untex)
        p, r, f = boundary_fscore(pred, pair.gt_flow, args.disc_thresh)
        p_sum += p; r_sum += r; f_sum += f; n += 1
        fscore_per_seq[pair.seq].append(f)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1:4d}/{ds.count()}] {time.time()-t0:5.1f}s")

    s = m.global_summary()
    by_seq = m.per_seq_epe_all()
    print(f"\n=== {Path(args.pred_root).name}  pass={args.pass_}  ({time.time()-t0:.0f}s) ===")
    for k in ("epe/all", "epe/matched", "epe/unmatched",
              "epe/s0_10", "epe/s10_40", "epe/s40+",
              "epe/disc", "epe/untex",
              "bad1/all", "bad3/all", "bad5/all"):
        if k in s:
            print(f"  {k:18s} {s[k]:.4f}")
    print(f"  boundary F1        {f_sum/n:.4f}   (P={p_sum/n:.4f}  R={r_sum/n:.4f})")
    print(f"  thresholds: disc={args.disc_thresh}  untex={args.untex_thresh}  tol_px=2")
    print("  per-sequence EPE/all, F1:")
    for seq in sorted(by_seq):
        seq_f1 = float(np.mean(fscore_per_seq[seq]))
        print(f"    {seq:14s} EPE={by_seq[seq]:.4f}  F1={seq_f1:.4f}")


if __name__ == "__main__":
    main()
