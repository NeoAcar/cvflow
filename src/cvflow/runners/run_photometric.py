"""Step 15: photometric-residual analysis on Sintel clean vs final.

Per pair: warp I1 by GT flow, residual = |I1 - I2_warped| on non-occluded GT.
Reports per-sequence mean residual and the global Clean->Final delta.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from cvflow.datasets.sintel import Sintel
from cvflow.masks.photometric import photometric_residual


def per_sequence_residual(pass_: str, sintel_root: str) -> dict[str, float]:
    ds = Sintel(sintel_root, pass_=pass_)
    sums = defaultdict(float)
    counts = defaultdict(int)
    for p in ds.pairs():
        r = photometric_residual(p.img1, p.img2, p.gt_flow)
        valid = (p.invalid == 0) & (p.occlusion == 0)
        if not valid.any():
            continue
        sums[p.seq] += float(r[valid].sum())
        counts[p.seq] += int(valid.sum())
    return {s: sums[s] / counts[s] for s in sums}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    args = ap.parse_args()

    t0 = time.time()
    clean = per_sequence_residual("clean", args.sintel_root)
    final = per_sequence_residual("final", args.sintel_root)
    print(f"elapsed: {time.time()-t0:.0f}s")

    print(f"\n{'seq':14s} {'clean':>8s} {'final':>8s} {'Δ':>8s}")
    for seq in sorted(clean):
        c, f = clean[seq], final[seq]
        print(f"{seq:14s} {c:8.3f} {f:8.3f} {f-c:+8.3f}")
    cm = float(np.mean(list(clean.values())))
    fm = float(np.mean(list(final.values())))
    print(f"\nmean across sequences: clean={cm:.3f}  final={fm:.3f}  Δ={fm-cm:+.3f}")


if __name__ == "__main__":
    main()
