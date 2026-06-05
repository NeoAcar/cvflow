"""Offline evaluation: load saved .npy predictions + Sintel GT, compute the full
mask suite (All/Matched/Unmatched/s0_10/s10_40/s40+/s60+/Disc/Untex/Blur) plus
EPE/SD/AE/Bad-1,3,5,10/A50,75,95 per mask and boundary F-score.

No GPU required.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cvflow.datasets.sintel import Sintel
from cvflow.masks.blur import blur_mask
from cvflow.masks.motion_boundary import disc_mask
from cvflow.masks.textureless import untext_mask
from cvflow.metrics.boundary_fscore import boundary_fscore
from cvflow.metrics.sintel import SintelMetrics


_MASKS = ("all", "matched", "unmatched", "s0_1", "s0_10", "s10_40", "s40+", "s60+",
          "disc", "untex", "blur")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--pass", dest="pass_", choices=["clean", "final"], default="clean")
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--disc-thresh", type=float, default=1.0)
    ap.add_argument("--untex-thresh", type=float, default=5.0)
    ap.add_argument("--blur-window", type=int, default=7)
    ap.add_argument("--blur-thresh", type=float, default=20.0)
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="If given, restrict evaluation to these sequence names "
                         "(e.g. --seqs alley_1 market_2). Default: all 23 sequences.")
    ap.add_argument("--dump-json", default=None,
                    help="If set, write per-sequence per-mask raw sums + counts to this JSON path "
                         "(input format for bootstrap_compare).")
    args = ap.parse_args()

    pred_dir = Path(args.pred_root) / "sintel" / args.pass_
    ds = Sintel(args.sintel_root, pass_=args.pass_)
    m = SintelMetrics()
    fscore_per_seq: dict[str, list[float]] = defaultdict(list)
    f_sum, p_sum, r_sum, n = 0.0, 0.0, 0.0, 0

    t0 = time.time()
    pair_iter = ds.pairs(seqs=args.seqs)
    n_total = ds.count() if args.seqs is None else sum(
        len(list((ds.root / args.pass_ / s).glob("frame_*.png"))) - 1 for s in args.seqs)
    for i, pair in enumerate(pair_iter):
        npy = pred_dir / pair.seq / f"frame_{pair.idx:04d}.npy"
        pred = np.load(npy)
        disc = disc_mask(pair.gt_flow, args.disc_thresh)
        untex = untext_mask(pair.img1, args.untex_thresh)
        blur = blur_mask(pair.img1, args.blur_window, args.blur_thresh)
        m.update(pred, pair.gt_flow, pair.occlusion, pair.invalid, pair.seq,
                 disc=disc, untex=untex, blur=blur)
        p, r, f = boundary_fscore(pred, pair.gt_flow, args.disc_thresh)
        p_sum += p; r_sum += r; f_sum += f; n += 1
        fscore_per_seq[pair.seq].append(f)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1:4d}/{n_total}] {time.time()-t0:5.1f}s")

    s = m.global_summary()
    by_seq = m.per_seq_epe_all()
    print(f"\n=== {Path(args.pred_root).name}  pass={args.pass_}  ({time.time()-t0:.0f}s) ===")
    print(f"  thresholds: disc={args.disc_thresh}  untex={args.untex_thresh}  "
          f"blur(window={args.blur_window},var<{args.blur_thresh})  tol_px=2")
    print()
    header = ("mask", "EPE", "SD", "AE°", "nEPE", "Bad1", "Bad3", "Bad5", "Bad10",
              "A50", "A75", "A95", "PearsonAE", "SpearAE")
    print(f"  {header[0]:>10s}  " + "  ".join(f"{h:>9s}" for h in header[1:]))
    for k in _MASKS:
        if f"epe/{k}" not in s:
            continue
        row = [k] + [s[f"{fld}/{k}"] for fld in
                     ("epe","sd","ae","nepe","bad1","bad3","bad5","bad10",
                      "A50","A75","A95","pearson","spearman")]
        cells = [f"{row[0]:>10s}"] + [f"{x:>9.4f}" for x in row[1:]]
        print("  " + "  ".join(cells))
    print(f"  {'boundary F1':>10s}  P={p_sum/n:.4f}  R={r_sum/n:.4f}  F={f_sum/n:.4f}")
    print()
    print("  per-sequence EPE/all, boundary F1:")
    for seq in sorted(by_seq):
        seq_f1 = float(np.mean(fscore_per_seq[seq]))
        print(f"    {seq:14s} EPE={by_seq[seq]:.4f}  F1={seq_f1:.4f}")

    if args.dump_json:
        m.dump_per_seq(args.dump_json)
        print(f"\nwrote per-seq stats to {args.dump_json}")


if __name__ == "__main__":
    main()
