"""H3 sensitivity: boundary F1 at gradient thresholds {0.5, 1.0, 2.0}.

If RAFT's F1 lead over GMFlow holds at all three thresholds, H3 is robust.
If it disappears or flips at one of them, the metric is biting its own tail.
Reads saved predictions; recomputes the F-score per pair using
`cvflow.metrics.boundary_fscore.boundary_fscore` at each threshold.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from cvflow.datasets.sintel import Sintel
from cvflow.metrics.boundary_fscore import boundary_fscore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=[
        "raft-raft-things-iter32",
        "gmflow-gmflow_things-e9887eda",
        "gmflow-gmflow_with_refine_things-36579974",
    ])
    ap.add_argument("--passes", nargs="+", default=["clean", "final"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--out", default="results/figures/boundary_threshold")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for tag in args.tags:
        for pass_ in args.passes:
            pred_root = Path("results/predictions") / tag / "sintel" / pass_
            if not pred_root.exists():
                print(f"  skip {tag} {pass_} — {pred_root} missing")
                continue
            ds = Sintel(args.sintel_root, pass_=pass_)
            print(f"  {tag} / {pass_} ...", flush=True)
            t0 = time.time()
            p_sum = {t: 0.0 for t in args.thresholds}
            r_sum = {t: 0.0 for t in args.thresholds}
            f_sum = {t: 0.0 for t in args.thresholds}
            n = 0
            for pair in ds.pairs():
                pred_path = pred_root / pair.seq / f"frame_{pair.idx:04d}.npy"
                pred = np.load(pred_path)
                for tau in args.thresholds:
                    p, r, f = boundary_fscore(pred, pair.gt_flow, grad_thresh=tau)
                    p_sum[tau] += p; r_sum[tau] += r; f_sum[tau] += f
                n += 1
            elapsed = time.time() - t0
            for tau in args.thresholds:
                rows.append({
                    "tag": tag, "pass": pass_, "threshold": tau,
                    "precision": p_sum[tau] / n,
                    "recall":    r_sum[tau] / n,
                    "F1":        f_sum[tau] / n,
                    "n_pairs":   n,
                })
                print(f"     τ={tau:>3.1f}  P={p_sum[tau]/n:.4f}  "
                      f"R={r_sum[tau]/n:.4f}  F1={f_sum[tau]/n:.4f}  ({elapsed:.0f}s total)")

    csv_path = out / "sensitivity.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {csv_path}")

        # Compact pivot print: F1 only, tag × (pass, τ)
        print("\nF1 summary:")
        configs = sorted({(r["tag"], r["pass"]) for r in rows})
        taus = sorted({r["threshold"] for r in rows})
        hdr = ["config"] + [f"τ={t}" for t in taus]
        print("  " + "  ".join(f"{h:>30s}" if i == 0 else f"{h:>8s}" for i, h in enumerate(hdr)))
        for tag, pass_ in configs:
            cells = [f"{tag} / {pass_}"]
            for t in taus:
                f1 = next(r["F1"] for r in rows if r["tag"] == tag and r["pass"] == pass_ and r["threshold"] == t)
                cells.append(f"{f1:.4f}")
            print("  " + "  ".join(f"{c:>30s}" if i == 0 else f"{c:>8s}" for i, c in enumerate(cells)))


if __name__ == "__main__":
    main()
