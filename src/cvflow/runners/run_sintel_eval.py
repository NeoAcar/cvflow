"""Step 8 runner: predict on Sintel, save predictions, accumulate per-mask metrics.

Targets for GMFlow basic on Sintel clean (from gmflow/scripts/evaluate.sh):
    EPE 1.495 | s0_10 0.457 s10_40 1.770 s40+ 8.257 | 1px 0.161 3px 0.059 5px 0.040
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cvflow.datasets.sintel import Sintel
from cvflow.metrics.sintel import SintelMetrics


GMFLOW_REFINE_PRESET = dict(
    padding_factor=32,
    attn_splits_list=[2, 8],
    corr_radius_list=[-1, 4],
    prop_radius_list=[-1, 1],
    num_scales=2,
    upsample_factor=4,
)


def load_model(kind: str, ckpt: str | None, raft_iters: int, gmflow_refine: bool = False):
    if kind == "raft":
        from cvflow.models.raft_wrapper import RaftWrapper
        ckpt = ckpt or "RAFT/RAFT/models/raft-things.pth"
        return RaftWrapper(ckpt, iters=raft_iters)
    if kind == "gmflow":
        from cvflow.models.gmflow_wrapper import GMFlowWrapper
        if gmflow_refine:
            ckpt = ckpt or "gmflow/gmflow/pretrained/gmflow_with_refine_things-36579974.pth"
            return GMFlowWrapper(ckpt, **GMFLOW_REFINE_PRESET)
        ckpt = ckpt or "gmflow/gmflow/pretrained/gmflow_things-e9887eda.pth"
        return GMFlowWrapper(ckpt)
    raise ValueError(kind)


def run(model_kind: str, ckpt: str | None, pass_: str, sintel_root: str, out_root: str,
        raft_iters: int, save: bool, gmflow_refine: bool = False, log_every: int = 100) -> dict:
    model = load_model(model_kind, ckpt, raft_iters, gmflow_refine=gmflow_refine)
    tag = model.name
    print(f"model {tag}  device={model.device}  pass={pass_}")

    ds = Sintel(sintel_root, pass_=pass_)
    metrics = SintelMetrics()
    pred_root = Path(out_root) / "predictions" / tag / "sintel" / pass_

    t0 = time.time()
    for i, p in enumerate(ds.pairs()):
        flow = model.predict(p.img1, p.img2)
        metrics.update(flow, p.gt_flow, p.occlusion, p.invalid, p.seq)
        if save:
            d = pred_root / p.seq
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"frame_{p.idx:04d}.npy", flow)
        if (i + 1) % log_every == 0:
            s = metrics.global_summary()
            print(f"  [{i+1:4d}/{ds.count()}] {time.time()-t0:6.1f}s  "
                  f"epe/all={s['epe/all']:.4f}  bad1={s['bad1/all']:.3f}")

    elapsed = time.time() - t0
    summary = metrics.global_summary()
    summary["_elapsed_s"] = elapsed
    summary["_tag"] = tag
    summary["_pass"] = pass_
    return {"summary": summary, "per_seq": metrics.per_seq_epe_all()}


def print_report(name: str, r: dict, targets: dict | None = None) -> None:
    s = r["summary"]
    print(f"\n=== {name}  pass={s['_pass']}  ({s['_elapsed_s']:.0f}s) ===")
    header = ("epe/all", "epe/matched", "epe/unmatched", "epe/s0_10", "epe/s10_40", "epe/s40+",
              "bad1/all", "bad3/all", "bad5/all")
    for k in header:
        v = s[k]
        tgt = (targets or {}).get(k)
        if tgt is None:
            print(f"  {k:18s} {v:.4f}")
        else:
            dpct = (v - tgt) / tgt * 100
            tag = "OK" if abs(dpct) <= 10 else "MISS"
            print(f"  {k:18s} {v:.4f}   target {tgt:.4f}   Δ {dpct:+.1f}%   {tag}")
    print("  per-sequence EPE/all:")
    for seq, e in sorted(r["per_seq"].items()):
        print(f"    {seq:14s} {e:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["raft", "gmflow"], required=True)
    ap.add_argument("--pass", dest="pass_", choices=["clean", "final"], default="clean")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--raft-iters", type=int, default=32)
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--gmflow-refine", action="store_true",
                    help="Use gmflow_with_refine_things checkpoint + refine config preset.")
    args = ap.parse_args()

    r = run(args.model, args.ckpt, args.pass_, args.sintel_root, args.out,
            raft_iters=args.raft_iters, save=not args.no_save,
            gmflow_refine=args.gmflow_refine)

    # GMFlow basic & refine, Sintel — official evaluate.sh numbers
    GMFLOW_TGTS = {
        ("basic",  "clean"): {"epe/all": 1.495, "epe/s0_10": 0.457, "epe/s10_40": 1.770, "epe/s40+": 8.257,
                              "bad1/all": 0.161, "bad3/all": 0.059, "bad5/all": 0.040},
        ("basic",  "final"): {"epe/all": 2.955, "epe/s0_10": 0.725, "epe/s10_40": 3.446, "epe/s40+": 17.701,
                              "bad1/all": 0.209, "bad3/all": 0.098, "bad5/all": 0.071},
        ("refine", "clean"): {"epe/all": 1.084, "epe/s0_10": 0.303, "epe/s10_40": 1.252, "epe/s40+": 6.261,
                              "bad1/all": 0.092, "bad3/all": 0.040, "bad5/all": 0.028},
        ("refine", "final"): {"epe/all": 2.475, "epe/s0_10": 0.511, "epe/s10_40": 2.810, "epe/s40+": 15.669,
                              "bad1/all": 0.147, "bad3/all": 0.077, "bad5/all": 0.058},
    }
    tgt = None
    if args.model == "gmflow":
        tgt = GMFLOW_TGTS[("refine" if args.gmflow_refine else "basic", args.pass_)]
    print_report(args.model, r, tgt)


if __name__ == "__main__":
    main()
