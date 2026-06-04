"""Compute and cache per-mask AE↔EPE correlations for each (model, pass).

Reads saved .npy predictions + Sintel GT, builds the same 11-mask set as
SintelMetrics, and computes Pearson + Spearman correlation between per-pixel
EPE and AE within each mask. Output is a JSON keyed by
"{model}_{pass}" → {mask: {pearson, spearman, n_samples}}.

We sample at most _SAMPLE_CAP paired pixels per mask (matches the reservoir
inside SintelMetrics) so the correlation reflects the same statistic the
SintelMetrics global_summary reports.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cvflow.datasets.sintel import Sintel
from cvflow.masks.blur import blur_mask
from cvflow.masks.motion_boundary import disc_mask
from cvflow.masks.textureless import untext_mask
from cvflow.metrics.middlebury import angular_error_deg


_SAMPLE_CAP = 5_000_000
_MASKS = ("all", "matched", "unmatched", "s0_1", "s0_10", "s10_40", "s40+", "s60+",
          "disc", "untex", "blur")

_TAGS = {
    "raft":          "raft-raft-things-iter32",
    "gmflow_basic":  "gmflow-gmflow_things-e9887eda",
    "gmflow_refine": "gmflow-gmflow_with_refine_things-36579974",
}


def correlations_for_corner(tag: str, pass_: str, sintel_root: str) -> dict:
    ds = Sintel(sintel_root, pass_=pass_)
    pred_root = Path("results/predictions") / tag / "sintel" / pass_
    epe_pool: dict[str, list[np.ndarray]] = {m: [] for m in _MASKS}
    ae_pool:  dict[str, list[np.ndarray]] = {m: [] for m in _MASKS}
    pool_size: dict[str, int] = {m: 0 for m in _MASKS}

    for pair in ds.pairs():
        pred = np.load(pred_root / pair.seq / f"frame_{pair.idx:04d}.npy")
        epe = np.linalg.norm(pred - pair.gt_flow, axis=-1)
        ae = angular_error_deg(pred, pair.gt_flow)
        valid = pair.invalid == 0
        speed = np.linalg.norm(pair.gt_flow, axis=-1)
        disc = disc_mask(pair.gt_flow)
        untex = untext_mask(pair.img1)
        blur = blur_mask(pair.img1)
        masks = {
            "all":       valid,
            "matched":   valid & (pair.occlusion == 0),
            "unmatched": valid & (pair.occlusion == 255),
            "s0_1":      valid & (speed < 1),
            "s0_10":     valid & (speed < 10),
            "s10_40":    valid & (speed >= 10) & (speed < 40),
            "s40+":      valid & (speed >= 40),
            "s60+":      valid & (speed >= 60),
            "disc":      valid & disc,
            "untex":     valid & untex,
            "blur":      valid & blur,
        }
        for m, mask in masks.items():
            if not mask.any() or pool_size[m] >= _SAMPLE_CAP:
                continue
            e = epe[mask]
            a = ae[mask]
            remaining = _SAMPLE_CAP - pool_size[m]
            take = min(remaining, e.size)
            epe_pool[m].append(e[:take].astype(np.float32))
            ae_pool[m].append(a[:take].astype(np.float32))
            pool_size[m] += take

    out: dict[str, dict] = {}
    for m in _MASKS:
        if not epe_pool[m]:
            out[m] = {"pearson": float("nan"), "spearman": float("nan"), "n": 0}
            continue
        e = np.concatenate(epe_pool[m]).astype(np.float64)
        a = np.concatenate(ae_pool[m]).astype(np.float64)
        if e.size < 2 or e.std() == 0 or a.std() == 0:
            out[m] = {"pearson": float("nan"), "spearman": float("nan"), "n": int(e.size)}
            continue
        out[m] = {
            "pearson":  float(np.corrcoef(e, a)[0, 1]),
            "spearman": float(spearmanr(e, a).statistic),
            "n":        int(e.size),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--out", default="results/correlations/sintel_per_mask.json")
    args = ap.parse_args()

    data: dict[str, dict] = {}
    for key, tag in _TAGS.items():
        for pass_ in ("clean", "final"):
            print(f"  computing {key} / {pass_} ...", flush=True)
            t0 = time.time()
            data[f"{key}_{pass_}"] = correlations_for_corner(tag, pass_, args.sintel_root)
            print(f"    done in {time.time()-t0:.0f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
