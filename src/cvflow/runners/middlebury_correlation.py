"""Per-sequence + global AE/EPE correlation on Middlebury (Pearson + Spearman).

Reads saved `.npy` predictions, applies the Middlebury invalid-GT sentinel
(`|u|<1e9 ∧ |v|<1e9`), and computes correlations on the valid pixels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cvflow.datasets.middlebury import Middlebury
from cvflow.metrics.middlebury import angular_error_deg, gt_valid_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--middlebury-root", default="datasets/Middleburry")
    ap.add_argument("--configs", nargs="+", default=[
        "raft-raft-things-iter32",
        "gmflow-gmflow_things-e9887eda",
    ])
    args = ap.parse_args()

    ds = Middlebury(args.middlebury_root)

    for tag in args.configs:
        print(f"\n=== {tag} (Middlebury) ===")
        all_e: list[np.ndarray] = []
        all_a: list[np.ndarray] = []
        print(f"  {'sequence':<14s}  {'n':>8s}  {'pearson':>8s}  {'spearman':>8s}")
        for p in ds.pairs():
            pred = np.load(Path("results/predictions") / tag / "middlebury" / p.seq / "flow10.npy")
            valid = gt_valid_mask(p.gt_flow)
            e = np.linalg.norm(pred - p.gt_flow, axis=-1)[valid].astype(np.float64)
            a = angular_error_deg(pred, p.gt_flow)[valid].astype(np.float64)
            pe = float(np.corrcoef(e, a)[0, 1]) if e.std() > 0 and a.std() > 0 else float("nan")
            sp = float(spearmanr(e, a).statistic)
            print(f"  {p.seq:<14s}  {e.size:>8d}  {pe:>8.4f}  {sp:>8.4f}")
            all_e.append(e); all_a.append(a)
        E = np.concatenate(all_e); A = np.concatenate(all_a)
        pe = float(np.corrcoef(E, A)[0, 1]) if E.std() > 0 and A.std() > 0 else float("nan")
        sp = float(spearmanr(E, A).statistic)
        print(f"  {'GLOBAL':<14s}  {E.size:>8d}  {pe:>8.4f}  {sp:>8.4f}")


if __name__ == "__main__":
    main()
