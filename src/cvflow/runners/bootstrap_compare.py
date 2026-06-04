"""Bootstrap-compare two model+pass configs across a fixed set of masks/metrics.

Reads per-sequence JSON dumps produced by `eval_from_saved --dump-json`,
computes paired-sequence-bootstrap CI on Δ = (b − a) for each (mask, metric)
combination, prints a table marking close calls (CI crosses zero).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cvflow.analysis.bootstrap import load_per_seq, paired_bootstrap


_DEFAULT_MASKS = ("all", "matched", "unmatched", "s0_1", "s0_10", "s10_40",
                  "s40+", "s60+", "disc", "untex", "blur")
_DEFAULT_METRICS = ("epe", "ae", "bad1", "bad10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Per-seq JSON for config A (e.g. results/per_seq_stats/raft_clean.json)")
    ap.add_argument("--b", required=True, help="Per-seq JSON for config B (e.g. results/per_seq_stats/gmflow_clean.json)")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--masks", nargs="+", default=list(_DEFAULT_MASKS))
    ap.add_argument("--metrics", nargs="+", default=list(_DEFAULT_METRICS))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = load_per_seq(args.a)
    b = load_per_seq(args.b)
    la = args.label_a or Path(args.a).stem
    lb = args.label_b or Path(args.b).stem

    print(f"\nPaired sequence bootstrap (n={args.n_boot})")
    print(f"  A = {la}  ({args.a})")
    print(f"  B = {lb}  ({args.b})")
    print()

    for metric in args.metrics:
        print(f"=== metric: {metric}  →  Δ = B − A ===")
        print(f"  {'mask':>10s}  {'A':>9s}  {'B':>9s}  {'ΔB-A':>9s}  {'95% CI':>20s}  {'p':>7s}  {'verdict':>10s}")
        for mask in args.masks:
            if mask not in a or mask not in b:
                continue
            r = paired_bootstrap(a, b, mask, metric, n_boot=args.n_boot, seed=args.seed)
            ci_str = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
            verdict = "NULL" if r.crosses_zero else ("A wins" if r.diff > 0 else "B wins")
            print(f"  {mask:>10s}  {r.mean_a:>9.4f}  {r.mean_b:>9.4f}  {r.diff:>+9.4f}  "
                  f"{ci_str:>20s}  {r.p_value:>7.4f}  {verdict:>10s}")
        print()


if __name__ == "__main__":
    main()
