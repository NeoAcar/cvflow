"""Step 4 follow-up: latency + peak VRAM as a function of input resolution.

Upsamples Sintel pairs (1024x436) by factors {1.0, 1.5, 2.0, 2.5} and reports
per-pair latency and peak VRAM for RAFT and GMFlow. Designed to fire (or fail
to fire) hypothesis 9 — GMFlow's O(N^2) attention is supposed to blow up.
"""

from __future__ import annotations

import argparse
from itertools import islice

import cv2
import numpy as np
import torch

from cvflow.datasets.sintel import Sintel


def upsample(img: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return img
    h, w = img.shape[:2]
    nh, nw = int(round(h * factor)), int(round(w * factor))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def measure(model, pairs, n: int = 10, warmup: int = 3) -> dict:
    device = model.device
    for i in range(warmup):
        model.predict(pairs[i % len(pairs)].img1, pairs[i % len(pairs)].img2)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    se, ee = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts: list[float] = []
    for i in range(n):
        p = pairs[i % len(pairs)]
        torch.cuda.synchronize(device)
        se.record()
        model.predict(p.img1, p.img2)
        ee.record()
        torch.cuda.synchronize(device)
        ts.append(se.elapsed_time(ee))

    return {
        "median_ms":  float(np.median(ts)),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    ds = Sintel(args.sintel_root, pass_="clean")
    raw = list(islice(ds.pairs(seqs=["alley_1", "market_2"]), 20))

    print(f"{'factor':>7}  {'WxH':>13}  {'RAFT ms':>10}  {'RAFT MB':>10}  {'GMFlow ms':>11}  {'GMFlow MB':>11}")
    for f in args.factors:
        sample = type(raw[0])(
            seq=raw[0].seq, idx=raw[0].idx,
            img1=upsample(raw[0].img1, f),
            img2=upsample(raw[0].img2, f),
            gt_flow=raw[0].gt_flow, occlusion=raw[0].occlusion, invalid=raw[0].invalid,
        )
        pairs = []
        for r in raw:
            pairs.append(type(r)(
                seq=r.seq, idx=r.idx,
                img1=upsample(r.img1, f), img2=upsample(r.img2, f),
                gt_flow=r.gt_flow, occlusion=r.occlusion, invalid=r.invalid,
            ))
        h, w = pairs[0].img1.shape[:2]

        from cvflow.models.raft_wrapper import RaftWrapper
        raft = RaftWrapper("RAFT/RAFT/models/raft-things.pth", iters=32)
        try:
            r = measure(raft, pairs, args.n)
            raft_ms, raft_mb = r["median_ms"], r["peak_vram_mb"]
        except torch.cuda.OutOfMemoryError:
            raft_ms, raft_mb = float("nan"), float("nan")
        del raft
        torch.cuda.empty_cache()

        from cvflow.models.gmflow_wrapper import GMFlowWrapper
        gm = GMFlowWrapper("gmflow/gmflow/pretrained/gmflow_things-e9887eda.pth")
        try:
            r = measure(gm, pairs, args.n)
            gm_ms, gm_mb = r["median_ms"], r["peak_vram_mb"]
        except torch.cuda.OutOfMemoryError:
            gm_ms, gm_mb = float("nan"), float("nan")
        del gm
        torch.cuda.empty_cache()

        print(f"{f:>7.2f}  {w:>5}x{h:<6}  "
              f"{raft_ms:>10.1f}  {raft_mb:>10.0f}  "
              f"{gm_ms:>11.1f}  {gm_mb:>11.0f}")


if __name__ == "__main__":
    main()
