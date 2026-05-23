"""Step 13: RAFT iteration sweep on representative sequences (alley_1, market_2).

Produces the EPE-vs-latency curve for hypothesis 6 — GMFlow is a single point
on the same plot. Limits inference to two sequences for tractability.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from cvflow.datasets.sintel import Sintel
from cvflow.metrics.sintel import SintelMetrics
from cvflow.models.raft_wrapper import RaftWrapper


def evaluate(iters: int, seqs: list[str], sintel_root: str) -> dict:
    model = RaftWrapper("RAFT/RAFT/models/raft-things.pth", iters=iters)
    ds = Sintel(sintel_root, pass_="clean")
    m = SintelMetrics()
    ts: list[float] = []

    # Warm-up
    pair0 = next(ds.pairs(seqs=[seqs[0]]))
    for _ in range(3):
        model.predict(pair0.img1, pair0.img2)
    torch.cuda.synchronize(model.device)

    se, ee = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    n_pairs = 0
    for p in ds.pairs(seqs=seqs):
        torch.cuda.synchronize(model.device)
        se.record()
        flow = model.predict(p.img1, p.img2)
        ee.record()
        torch.cuda.synchronize(model.device)
        ts.append(se.elapsed_time(ee))
        m.update(flow, p.gt_flow, p.occlusion, p.invalid, p.seq)
        n_pairs += 1

    summary = m.global_summary()
    del model
    torch.cuda.empty_cache()
    return {
        "iters": iters,
        "n_pairs": n_pairs,
        "epe_all":      summary["epe/all"],
        "epe_s0_10":    summary["epe/s0_10"],
        "epe_s10_40":   summary["epe/s10_40"],
        "epe_s40+":     summary["epe/s40+"],
        "median_ms":    float(np.median(ts)),
        "mean_ms":      float(np.mean(ts)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["alley_1", "market_2"])
    ap.add_argument("--iters", nargs="+", type=int, default=[4, 8, 12, 32])
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    args = ap.parse_args()

    print(f"sequences: {args.seqs}")
    print(f"iter sweep: {args.iters}")
    print()
    print(f"{'iters':>6} {'EPE/all':>9} {'s0_10':>8} {'s10_40':>8} {'s40+':>8} {'med ms':>8}")
    t0 = time.time()
    for it in args.iters:
        r = evaluate(it, args.seqs, args.sintel_root)
        print(f"{r['iters']:>6} {r['epe_all']:>9.4f} {r['epe_s0_10']:>8.4f} "
              f"{r['epe_s10_40']:>8.4f} {r['epe_s40+']:>8.4f} {r['median_ms']:>8.1f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
