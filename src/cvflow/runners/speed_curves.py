"""Speed-vs-metric line plots on Sintel.

Bins pixels by GT-flow magnitude using fine-grained edges and plots:
  - EPE
  - AE (degrees)
  - normalized EPE = EPE / |gt_flow|  (only on pixels with |gt| > 1)
  - Bad-1 fraction
as line plots, one curve per (model, pass). Writes PNGs to
results/figures/speed_curves/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvflow.datasets.sintel import Sintel
from cvflow.metrics.middlebury import angular_error_deg


_EDGES = np.array([0.0, 1.0, 3.0, 10.0, 20.0, 40.0, 60.0, np.inf])
_LABELS = ["s0-1", "s1-3", "s3-10", "s10-20", "s20-40", "s40-60", "s60+"]
_NEPE_FLOOR = 1.0


def accumulate(pred_root: Path, sintel_root: str, pass_: str):
    """Stream over the dataset, bin pixels, return sums and counts per bucket."""
    ds = Sintel(sintel_root, pass_=pass_)
    nb = len(_LABELS)
    sum_epe   = np.zeros(nb, dtype=np.float64)
    sum_ae    = np.zeros(nb, dtype=np.float64)
    sum_nepe  = np.zeros(nb, dtype=np.float64)
    nepe_n    = np.zeros(nb, dtype=np.int64)
    sum_bad1  = np.zeros(nb, dtype=np.int64)
    cnt       = np.zeros(nb, dtype=np.int64)

    pred_dir = pred_root / "sintel" / pass_
    for p in ds.pairs():
        pred = np.load(pred_dir / p.seq / f"frame_{p.idx:04d}.npy")
        speed = np.linalg.norm(p.gt_flow, axis=-1)
        valid = p.invalid == 0
        epe = np.linalg.norm(pred - p.gt_flow, axis=-1)
        ae = angular_error_deg(pred, p.gt_flow)
        # bucket index per pixel: 0..nb-1
        idx = np.digitize(speed, _EDGES[1:-1])
        for b in range(nb):
            m = valid & (idx == b)
            if not m.any():
                continue
            e = epe[m]; a = ae[m]; s = speed[m]
            cnt[b]      += e.size
            sum_epe[b]  += float(e.sum())
            sum_ae[b]   += float(a.sum())
            sum_bad1[b] += int((e > 1).sum())
            big = s > _NEPE_FLOOR
            if big.any():
                sum_nepe[b] += float((e[big] / s[big]).sum())
                nepe_n[b]   += int(big.sum())

    safe_cnt = np.where(cnt == 0, 1, cnt)
    safe_nepe_n = np.where(nepe_n == 0, 1, nepe_n)
    return {
        "epe":   np.where(cnt > 0, sum_epe / safe_cnt, np.nan),
        "ae":    np.where(cnt > 0, sum_ae / safe_cnt, np.nan),
        "nepe":  np.where(nepe_n > 0, sum_nepe / safe_nepe_n, np.nan),
        "bad1":  np.where(cnt > 0, sum_bad1 / safe_cnt, np.nan),
        "count": cnt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--out", default="results/figures/speed_curves")
    ap.add_argument("--configs", nargs="+", default=[
        "raft-raft-things-iter32:clean",
        "raft-raft-things-iter32:final",
        "gmflow-gmflow_things-e9887eda:clean",
        "gmflow-gmflow_things-e9887eda:final",
    ])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()
    for spec in args.configs:
        tag, pass_ = spec.split(":")
        print(f"  binning {tag} {pass_} ...")
        results[spec] = accumulate(Path("results/predictions") / tag, args.sintel_root, pass_)

    print(f"\nbinned in {time.time()-t0:.0f}s. Pixel counts per bucket:")
    for spec, r in results.items():
        print(f"  {spec:60s} {r['count']}")

    # Plot — one PNG per metric, all 4 configs on the same axes
    x = np.arange(len(_LABELS))
    style = {
        "raft-raft-things-iter32:clean":      dict(color="C0", linestyle="-",  label="RAFT clean"),
        "raft-raft-things-iter32:final":      dict(color="C0", linestyle="--", label="RAFT final"),
        "gmflow-gmflow_things-e9887eda:clean":dict(color="C1", linestyle="-",  label="GMFlow clean"),
        "gmflow-gmflow_things-e9887eda:final":dict(color="C1", linestyle="--", label="GMFlow final"),
    }
    metrics = [
        ("epe",   "EPE (px)",               False),
        ("ae",    "AE (degrees)",           False),
        ("nepe",  "normalized EPE = EPE / |gt|  (|gt| > 1 only)", False),
        ("bad1",  "Bad-1 fraction",         False),
    ]
    for key, ylabel, ylog in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for spec, r in results.items():
            ax.plot(x, r[key], marker="o", **style.get(spec, {"label": spec}))
        ax.set_xticks(x); ax.set_xticklabels(_LABELS)
        ax.set_xlabel("GT-flow magnitude bucket (px)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  vs  speed bucket")
        if ylog:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = out / f"{key}_vs_speed.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  wrote {path}")

    # Also write the underlying CSV
    import csv
    with (out / "speed_curves.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "metric"] + _LABELS)
        for spec, r in results.items():
            for key, _, _ in metrics:
                w.writerow([spec, key] + [f"{v:.4f}" if not np.isnan(v) else "nan" for v in r[key]])
            w.writerow([spec, "count"] + list(map(int, r["count"])))
    print(f"  wrote {out / 'speed_curves.csv'}")


if __name__ == "__main__":
    main()
