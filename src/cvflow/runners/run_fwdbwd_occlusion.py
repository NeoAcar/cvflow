"""Forward-backward consistency derived occlusion mask (methodology §2.2 cross-check).

For each Sintel pair, predict forward flow f12 = model(I1, I2) and backward
flow f21 = model(I2, I1). A pixel x is consistent iff
    || f12(x) + f21(x + f12(x)) ||  <  alpha * (||f12(x)||^2 + ||f21(x+f12(x))||^2) + beta
following Sundaram et al. 2010 / Meister et al. 2018. Pixels failing the check
are derived occlusion candidates. We compare against the Sintel native
occlusion mask via IoU and pixel agreement.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cvflow.datasets.sintel import Sintel


def warp_flow(flow_back: np.ndarray, flow_fwd: np.ndarray) -> np.ndarray:
    """Sample flow_back at positions x + flow_fwd(x). HxWx2."""
    h, w = flow_fwd.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    map_x = xx + flow_fwd[..., 0]
    map_y = yy + flow_fwd[..., 1]
    import cv2
    sampled_x = cv2.remap(flow_back[..., 0], map_x, map_y,
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sampled_y = cv2.remap(flow_back[..., 1], map_x, map_y,
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.stack([sampled_x, sampled_y], axis=-1)


def fwdbwd_occlusion(f12: np.ndarray, f21: np.ndarray,
                     alpha: float = 0.01, beta: float = 0.5) -> np.ndarray:
    """Return bool HxW mask: True = inconsistent (derived occlusion candidate)."""
    f21_at = warp_flow(f21, f12)
    diff_sq = ((f12 + f21_at) ** 2).sum(axis=-1)
    norm_sq = (f12 ** 2).sum(axis=-1) + (f21_at ** 2).sum(axis=-1)
    return diff_sq > (alpha * norm_sq + beta)


def load_model(kind: str):
    if kind == "raft":
        from cvflow.models.raft_wrapper import RaftWrapper
        return RaftWrapper("RAFT/RAFT/models/raft-things.pth", iters=32)
    from cvflow.models.gmflow_wrapper import GMFlowWrapper
    return GMFlowWrapper("gmflow/gmflow/pretrained/gmflow_things-e9887eda.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["raft", "gmflow"], default="raft")
    ap.add_argument("--seqs", nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--fwd-cache", default=None,
                    help="Path to saved forward .npy root (e.g. results/predictions/<tag>/sintel/clean). "
                         "If given, skip forward inference and load from disk.")
    args = ap.parse_args()

    model = load_model(args.model)
    print(f"model={model.name}  alpha={args.alpha}  beta={args.beta}")
    if args.fwd_cache:
        print(f"fwd cache: {args.fwd_cache}")

    ds = Sintel(args.sintel_root, pass_="clean")
    seqs = args.seqs or ds.seqs

    # global confusion against Sintel native occlusion
    tp = fp = fn = tn = 0
    total_native_pos = 0
    total_derived_pos = 0
    total_valid = 0

    t0 = time.time()
    fwd_root = Path(args.fwd_cache) if args.fwd_cache else None
    for p in ds.pairs(seqs=seqs):
        if fwd_root is not None:
            f12 = np.load(fwd_root / p.seq / f"frame_{p.idx:04d}.npy")
        else:
            f12 = model.predict(p.img1, p.img2)
        f21 = model.predict(p.img2, p.img1)
        derived = fwdbwd_occlusion(f12, f21, args.alpha, args.beta)
        native = p.occlusion == 255
        valid = p.invalid == 0
        d = derived & valid
        n_ = native & valid
        tp += int((d & n_).sum())
        fp += int((d & ~n_ & valid).sum())
        fn += int((~d & n_ & valid).sum())
        tn += int((~d & ~n_ & valid).sum())
        total_native_pos += int(n_.sum())
        total_derived_pos += int(d.sum())
        total_valid += int(valid.sum())

    elapsed = time.time() - t0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    print(f"\n=== fwd-bwd derived occlusion vs Sintel native ({elapsed:.0f}s) ===")
    print(f"  native occluded pct:    {total_native_pos/total_valid*100:.2f}%")
    print(f"  derived occluded pct:   {total_derived_pos/total_valid*100:.2f}%")
    print(f"  precision:              {precision:.4f}")
    print(f"  recall:                 {recall:.4f}")
    print(f"  F1:                     {f1:.4f}")
    print(f"  IoU:                    {iou:.4f}")


if __name__ == "__main__":
    main()
