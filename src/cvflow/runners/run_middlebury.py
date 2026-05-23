"""Step 12: Cross-dataset Sintel-trained checkpoint -> Middlebury.

Both models use their Things-trained checkpoints (zero-shot to Middlebury) per
the methodology's cross-dataset framing (hypothesis 10).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cvflow.datasets.middlebury import Middlebury
from cvflow.metrics.middlebury import summary


def load_model(kind: str):
    if kind == "raft":
        from cvflow.models.raft_wrapper import RaftWrapper
        return RaftWrapper("RAFT/RAFT/models/raft-things.pth", iters=32)
    if kind == "gmflow":
        from cvflow.models.gmflow_wrapper import GMFlowWrapper
        return GMFlowWrapper("gmflow/gmflow/pretrained/gmflow_things-e9887eda.pth")
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["raft", "gmflow", "both"], default="both")
    ap.add_argument("--middlebury-root", default="datasets/Middleburry")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ds = Middlebury(args.middlebury_root)
    print(f"Middlebury: {len(ds.seqs)} sequences with GT")

    for kind in (["raft", "gmflow"] if args.model == "both" else [args.model]):
        print(f"\n=== {kind} (things-trained, zero-shot) ===")
        model = load_model(kind)
        pred_root = Path(args.out) / "predictions" / model.name / "middlebury"
        rows = []
        t0 = time.time()
        for p in ds.pairs():
            flow = model.predict(p.img1, p.img2)
            d = pred_root / p.seq
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "flow10.npy", flow)
            s = summary(flow, p.gt_flow)
            rows.append((p.seq, s))
            print(f"  {p.seq:14s} EE={s['ee_mean']:.3f}  AE={s['ae_mean']:.2f}°  "
                  f"R0.5={s['R0.5']:.3f} R1.0={s['R1.0']:.3f} R2.0={s['R2.0']:.3f}  "
                  f"A50={s['A50']:.3f} A95={s['A95']:.3f}")
        print(f"  mean EE: {np.mean([r['ee_mean'] for _,r in rows]):.4f}  "
              f"mean AE: {np.mean([r['ae_mean'] for _,r in rows]):.2f}°  "
              f"({time.time()-t0:.1f}s)")
        del model


if __name__ == "__main__":
    main()
