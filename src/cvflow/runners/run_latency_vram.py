"""Step 14: per-pair wall-clock latency + peak VRAM at 1024x436 (Sintel resolution).

Warm-up 5 pairs, time over the next N pairs with torch.cuda.Event.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from cvflow.datasets.sintel import Sintel


def measure(model, pairs, warmup: int = 5, n: int = 50) -> dict:
    device = model.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    for i in range(warmup):
        model.predict(pairs[i % len(pairs)].img1, pairs[i % len(pairs)].img2)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    timings_ms: list[float] = []

    for i in range(n):
        p = pairs[i % len(pairs)]
        torch.cuda.synchronize(device)
        start_evt.record()
        _ = model.predict(p.img1, p.img2)
        end_evt.record()
        torch.cuda.synchronize(device)
        timings_ms.append(start_evt.elapsed_time(end_evt))

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return {
        "n": n,
        "mean_ms": float(np.mean(timings_ms)),
        "median_ms": float(np.median(timings_ms)),
        "std_ms": float(np.std(timings_ms)),
        "min_ms": float(np.min(timings_ms)),
        "max_ms": float(np.max(timings_ms)),
        "peak_vram_mb": float(peak_mb),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — step 14 requires GPU.")

    ds = Sintel(args.sintel_root, pass_="clean")
    pairs = []
    for i, p in enumerate(ds.pairs(seqs=["alley_1", "market_2", "cave_4"])):
        pairs.append(p)
        if len(pairs) >= max(args.warmup + 5, args.n + args.warmup):
            break
    print(f"Loaded {len(pairs)} pairs for timing (mixed sequences)")

    print(f"\n=== RAFT (raft-things, 32 iters) ===")
    from cvflow.models.raft_wrapper import RaftWrapper
    raft = RaftWrapper("RAFT/RAFT/models/raft-things.pth", iters=32)
    r = measure(raft, pairs, args.warmup, args.n)
    print(f"  median={r['median_ms']:.1f}ms  mean={r['mean_ms']:.1f}±{r['std_ms']:.1f}ms  "
          f"[{r['min_ms']:.1f}–{r['max_ms']:.1f}]  peak_VRAM={r['peak_vram_mb']:.0f}MB")
    del raft
    torch.cuda.empty_cache()

    print(f"\n=== GMFlow (gmflow_things, basic) ===")
    from cvflow.models.gmflow_wrapper import GMFlowWrapper
    gm = GMFlowWrapper("gmflow/gmflow/pretrained/gmflow_things-e9887eda.pth")
    r = measure(gm, pairs, args.warmup, args.n)
    print(f"  median={r['median_ms']:.1f}ms  mean={r['mean_ms']:.1f}±{r['std_ms']:.1f}ms  "
          f"[{r['min_ms']:.1f}–{r['max_ms']:.1f}]  peak_VRAM={r['peak_vram_mb']:.0f}MB")


if __name__ == "__main__":
    main()
