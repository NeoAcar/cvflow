"""Step 7 sanity check: zero-shot Things-trained EPE on Sintel-train Clean.

Targets (methodology §4 item 13, ±10%):
    RAFT  raft-things.pth         clean 1.43
    GMFlow gmflow_things-e9887eda  clean ~1.50
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cvflow.datasets.sintel import Sintel
from cvflow.models.gmflow_wrapper import GMFlowWrapper
from cvflow.models.raft_wrapper import RaftWrapper


def epe_with_invalid(pred: np.ndarray, gt: np.ndarray, invalid: np.ndarray) -> tuple[float, int]:
    err = np.linalg.norm(pred - gt, axis=-1)
    valid = invalid == 0
    n = int(valid.sum())
    return float(err[valid].sum()), n


def run_model(model, ds: Sintel, log_every: int = 100) -> dict:
    seq_sum: dict[str, float] = defaultdict(float)
    seq_cnt: dict[str, int] = defaultdict(int)
    total_sum = 0.0
    total_cnt = 0
    t0 = time.time()
    for i, pair in enumerate(ds.pairs()):
        pred = model.predict(pair.img1, pair.img2)
        s, n = epe_with_invalid(pred, pair.gt_flow, pair.invalid)
        seq_sum[pair.seq] += s
        seq_cnt[pair.seq] += n
        total_sum += s
        total_cnt += n
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:4d}/1041] elapsed={elapsed:5.1f}s running_epe={total_sum/total_cnt:.4f}")
    return {
        "global_epe": total_sum / total_cnt,
        "per_seq_epe": {s: seq_sum[s] / seq_cnt[s] for s in seq_sum},
        "elapsed_s": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sintel-root", default="datasets/Sintel")
    parser.add_argument("--raft-ckpt", default="RAFT/RAFT/models/raft-things.pth")
    parser.add_argument("--gmflow-ckpt", default="gmflow/gmflow/pretrained/gmflow_things-e9887eda.pth")
    parser.add_argument("--raft-iters", type=int, default=32)
    parser.add_argument("--model", choices=["raft", "gmflow", "both"], default="both")
    args = parser.parse_args()

    ds = Sintel(args.sintel_root, pass_="clean")
    print(f"Sintel clean: {len(ds.seqs)} sequences, {ds.count()} pairs")

    if args.model in ("raft", "both"):
        print("\n=== RAFT (raft-things.pth, 32 iters) ===")
        raft = RaftWrapper(args.raft_ckpt, iters=args.raft_iters)
        print(f"device={raft.device}")
        r = run_model(raft, ds)
        print(f"\nRAFT global EPE: {r['global_epe']:.4f}  (target 1.43, tol ±10%)")
        print(f"elapsed: {r['elapsed_s']:.1f}s")
        print("per-sequence:")
        for s, e in sorted(r["per_seq_epe"].items()):
            print(f"  {s:14s} {e:.4f}")
        del raft

    if args.model in ("gmflow", "both"):
        print("\n=== GMFlow (gmflow_things-e9887eda.pth, no refinement) ===")
        gm = GMFlowWrapper(args.gmflow_ckpt)
        print(f"device={gm.device}")
        g = run_model(gm, ds)
        print(f"\nGMFlow global EPE: {g['global_epe']:.4f}  (target 1.495, tol ±10%)")
        print(f"elapsed: {g['elapsed_s']:.1f}s")
        print("per-sequence:")
        for s, e in sorted(g["per_seq_epe"].items()):
            print(f"  {s:14s} {e:.4f}")


if __name__ == "__main__":
    main()
