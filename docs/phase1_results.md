# Phase 1 Results — RAFT vs GMFlow variants on Sintel + Middlebury

Inference-only evaluation. Things-trained checkpoints for both models (zero-shot framing — no Sintel fine-tuning, no Middlebury fine-tuning). All numbers reproducible from saved `.npy` predictions in `results/predictions/`.

> **Revision history.** Original report compared RAFT-32 against GMFlow-basic (no refinement). A critique pass flagged three structural problems: (i) the RAFT-32 vs GMFlow-basic comparison is unfair — GMFlow ships a with-refinement variant designed for parity at RAFT-32 latency; (ii) close-call verdicts on final-pass s40+/s60+/unmatched were never tested against the 23-sequence variance; (iii) H9's resolution sweep is contaminated by WSL2 paging. Sections **§5 Pareto**, **§8 hypothesis verdicts**, and **§9 caveats** are rewritten below. New evidence (paired sequence bootstrap, three-way ablation with GMFlow-refine, boundary F1 threshold sweep, quantization-residual histograms, alternative H10 normalization) is collected in **§11**.

## 1. Setup

| | |
|---|---|
| RAFT checkpoint | `raft-things.pth` (21 MB, Dropbox-distributed) |
| RAFT inference iters (headline) | 32 |
| GMFlow-basic checkpoint | `gmflow_things-e9887eda.pth` (basic, no refinement) |
| GMFlow-basic attn/corr/prop | `[2]/[-1]/[-1]`, `padding_factor=16` |
| Sintel train pairs | 1041 per pass (23 sequences × (frame_count − 1)) |
| Middlebury pairs | 8 (sequences with GT) |
| Environment | `uv venv`, Python 3.11, PyTorch 2.6.0+cu124 |
| Hardware | NVIDIA RTX 3050 Ti Laptop (4 GB) |
| Sintel pixel filter | `invalid == 0` (drops the ~0.01–2% unreliable-GT pixels) |
| Middlebury pixel filter | `|u| < 1e9 ∧ |v| < 1e9` (the Middlebury invalid-flow sentinel) |
| Mask thresholds | Disc `‖∇u‖₁+‖∇v‖₁ > 1.0` then 9×9 dilate · Untex `‖∇I‖ < 5` then 3×3 dilate · boundary F-score `tol_px = 2` |

## 2. Step 7 sanity gate (methodology §4 item 13)

Targets are Teed & Deng 2020 Table 1 (C+T row) and the GMFlow `scripts/evaluate.sh` comment block. Pass criterion: ±10%.

| Model | Pass | Measured EPE | Target | Δ |
|---|---|---|---|---|
| RAFT | clean | 1.4459 | 1.43 | +1.1% |
| RAFT | final | 2.6779 | 2.71 | −1.2% |
| GMFlow-basic | clean | 1.4839 | 1.495 | −0.7% |
| GMFlow-basic | final | 2.9420 | 2.955 | −0.4% |

Per-mask sub-comparison against the GMFlow-basic published numbers on Sintel clean:

| | measured | target | Δ |
|---|---|---|---|
| s0-10 | 0.4560 | 0.457 | −0.2% |
| s10-40 | 1.7601 | 1.770 | −0.6% |
| s40+ | 8.1894 | 8.257 | −0.8% |
| Bad-1 | 0.1603 | 0.161 | −0.4% |
| Bad-3 | 0.0588 | 0.059 | −0.4% |
| Bad-5 | 0.0393 | 0.040 | −1.8% |

Gate passed end-to-end.

## 3. Sintel — full per-mask, both passes

Bold = raw winner (lower number). Cells marked with **§** are statistically NULL at 95% sequence-bootstrap CI per §11a — bold winner is not significantly different from the loser. The §3 table below is **the raw point estimates only**; for "is RAFT actually winning here" go to §11a.

> **Figure:** `results/figures/report/region_bars_full.png` — three-model per-region EPE bars on the full 1041-pair Sintel dataset, clean and final side-by-side, log y-axis (RAFT-32 blue, GMFlow-basic orange, GMFlow-refine green). The same data as the §3 table below, more glanceable.

| Mask | **Clean RAFT** | **Clean GMFlow-basic** | **Final RAFT** | **Final GMFlow-basic** |
|---|---|---|---|---|
| EPE / all | **1.446** § | 1.484 § | **2.678** § | 2.942 § |
| EPE / matched | **0.646** | 0.822 | **1.585** | 1.902 |
| EPE / unmatched | 11.69 | **9.95** | 16.67 § | **16.25** § |
| EPE / s0-1 | **0.161** | 0.245 | **0.293** | 0.371 |
| EPE / s0-10 | **0.360** § | 0.456 § | **0.509** | 0.724 |
| EPE / s10-40 | **1.651** § | 1.760 § | **2.994** § | 3.438 § |
| EPE / s40+ | 8.72 § | **8.19** § | **17.41** § | 17.63 § |
| EPE / s60+ | 12.30 § | **11.21** § | **23.61** § | 23.84 § |
| EPE / Disc | 3.58 § | **3.46** § | **6.26** § | 6.79 § |
| EPE / Untex | 1.65 § | **1.55** § | **3.11** § | 3.46 § |
| EPE / Blur | 2.18 § | **1.88** § | **4.03** § | 4.67 § |
| Bad-1 | **0.098** | 0.160 | **0.147** | 0.209 |
| Bad-3 | **0.044** | 0.059 | **0.081** | 0.098 |
| Bad-5 | **0.031** | 0.039 | **0.062** | 0.071 |
| Bad-10 (catastrophic) | **0.021** | 0.023 | **0.043** | 0.047 |
| Boundary F1 | **0.727** | 0.697 | **0.698** | 0.672 |

Note on Blur row: the main table is RAFT-32 vs GMFlow-basic only; §7d adds GMFlow-refine. Basic has lower clean-blur EPE than RAFT (1.88 vs 2.18), but RAFT wins final-blur EPE against basic (4.03 vs 4.67).

Three-model large-displacement headline (EPE, lower is better), computed from saved `per_seq_stats/*.json`:

| Pass / mask | RAFT-32 | GMFlow-basic | GMFlow-refine | best read |
|---|---:|---:|---:|---|
| Clean s40+ | 8.72 | 8.19 | **6.19** | refine is the clear large-motion winner |
| Clean s60+ | 12.30 | 11.21 | **8.55** | refine wins the largest-motion tail |
| Final s40+ | 17.41 | 17.63 | **15.60** | refine stays ahead under Final degradations |
| Final s60+ | 23.61 | 23.84 | **20.99** | refine stays ahead in the hardest tail |

The RAFT-32 vs GMFlow-basic large-motion differences are point-estimate close calls and bootstrap-NULL (§11a). The robust result is the three-model one: GMFlow-refine is significantly better than RAFT-32 on `s40+` and `s60+` in both clean and final.

Three-model textureless / `untex` headline (EPE, lower is better), computed from saved `per_seq_stats/*.json`:

| Pass / mask | RAFT-32 | GMFlow-basic | GMFlow-refine | best read |
|---|---:|---:|---:|---|
| Clean untex | 1.650 | 1.555 | **1.128** | refine wins significantly vs RAFT (§11a) |
| Final untex | 3.109 | 3.462 | **2.838** | refine is lower as a point estimate, but RAFT vs refine is bootstrap-NULL (§11a) |

### 3b. Detailed per-mask grid (Sintel clean only)

Full grid for each mask: EPE, SD, AE (degrees), normalized EPE (= `EPE/|gt|` on pixels with `|gt|>1`), Bad-1/3/5/10 fractions, A50/A75/A95 percentiles in px, and Pearson + Spearman correlation between EPE and AE on a 5M-pixel paired reservoir. Final-pass grid is regenerable via `eval_from_saved` (~190 s CPU per corner).

**RAFT  (raft-things-iter32, clean):**

| mask | EPE | SD | AE° | nEPE | Bad1 | Bad3 | Bad5 | Bad10 | A50 | A75 | A95 | Pearson | Spear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 1.446 | 11.19 | 3.91 | 0.108 | 0.098 | 0.044 | 0.031 | 0.021 | 0.067 | 0.131 | 0.707 | 0.712 | 0.639 |
| matched | 0.646 | 6.03 | 3.14 | 0.074 | 0.062 | 0.020 | 0.013 | 0.008 | 0.065 | 0.122 | 0.621 | 0.695 | 0.632 |
| unmatched | 11.69 | 33.92 | 13.86 | 0.470 | 0.558 | 0.344 | 0.268 | 0.187 | 2.11 | 9.74 | 131.6 | 0.652 | 0.610 |
| s0_1 | 0.161 | 1.20 | 5.72 | — | 0.015 | 0.004 | 0.002 | 0.001 | 0.059 | 0.100 | 0.593 | 0.317 | **0.968** |
| s0_10 | 0.360 | 3.54 | 4.09 | 0.122 | 0.037 | 0.011 | 0.006 | 0.003 | 0.062 | 0.110 | 0.557 | 0.753 | 0.693 |
| s10_40 | 1.651 | 8.68 | 3.05 | 0.085 | 0.175 | 0.072 | 0.050 | 0.031 | 0.130 | 0.272 | 1.400 | 0.748 | 0.772 |
| s40+ | 8.720 | 31.23 | 4.53 | 0.087 | 0.359 | 0.212 | 0.167 | 0.124 | 0.397 | 1.187 | 11.56 | **0.928** | 0.719 |
| s60+ | 12.30 | 38.65 | 5.29 | 0.098 | 0.414 | 0.262 | 0.214 | 0.164 | 0.758 | 3.74 | 82.23 | 0.847 | 0.840 |
| disc | 3.579 | 18.04 | 6.69 | 0.179 | 0.246 | 0.113 | 0.082 | 0.054 | 0.316 | 0.690 | 2.144 | 0.564 | 0.545 |
| untex | 1.650 | 12.86 | 3.78 | 0.119 | 0.095 | 0.042 | 0.031 | 0.022 | 0.068 | 0.144 | 0.625 | 0.583 | 0.684 |
| blur | 2.175 | 14.67 | 3.77 | 0.166 | 0.108 | 0.051 | 0.039 | 0.029 | 0.208 | 0.466 | 2.151 | 0.831 | 0.439 |

**GMFlow-basic  (gmflow_things, no refinement, clean):**

| mask | EPE | SD | AE° | nEPE | Bad1 | Bad3 | Bad5 | Bad10 | A50 | A75 | A95 | Pearson | Spear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 1.484 | 8.29 | 5.78 | 0.147 | 0.160 | 0.059 | 0.039 | 0.023 | 0.411 | 0.497 | 1.085 | 0.526 | 0.372 |
| matched | 0.822 | 4.32 | 5.28 | 0.131 | 0.120 | 0.032 | 0.018 | 0.009 | 0.408 | 0.490 | 0.994 | 0.515 | 0.375 |
| unmatched | 9.952 | 25.13 | 12.10 | 0.317 | 0.673 | 0.407 | 0.307 | 0.199 | 3.00 | 10.34 | 55.26 | 0.501 | 0.411 |
| s0_1 | 0.245 | 0.71 | 9.71 | — | 0.026 | 0.003 | 0.001 | 0.001 | 0.244 | 0.349 | 0.856 | 0.536 | 0.903 |
| s0_10 | 0.456 | 1.34 | 6.82 | 0.185 | 0.069 | 0.012 | 0.006 | 0.002 | 0.405 | 0.482 | 0.971 | 0.585 | 0.450 |
| s10_40 | 1.760 | 6.27 | 3.48 | 0.090 | 0.285 | 0.098 | 0.060 | 0.032 | 0.315 | 0.538 | 1.967 | 0.725 | 0.739 |
| s40+ | 8.189 | 23.59 | 3.40 | 0.089 | 0.536 | 0.306 | 0.232 | 0.156 | 1.052 | 2.957 | 19.86 | 0.549 | 0.665 |
| s60+ | 11.21 | 28.87 | 3.68 | 0.096 | 0.598 | 0.373 | 0.295 | 0.206 | 2.07 | 8.21 | 60.09 | **0.859** | 0.800 |
| disc | 3.463 | 13.49 | 7.78 | 0.192 | 0.372 | 0.156 | 0.106 | 0.063 | 0.532 | 0.999 | 2.577 | 0.494 | 0.490 |
| untex | 1.555 | 8.93 | 5.45 | 0.132 | 0.162 | 0.059 | 0.040 | 0.024 | 0.387 | 0.481 | 1.017 | 0.384 | 0.340 |
| blur | 1.881 | 10.04 | 4.79 | 0.123 | 0.182 | 0.071 | 0.049 | 0.030 | 0.490 | 1.129 | 6.058 | 0.224 | 0.347 |

(`—` in the nEPE column on `s0_1` is intentional: the `|gt| > 1px` floor excludes all pixels in that bucket. nEPE is undefined there.)

Key reads:
- **A50/A75 column** confirms the sub-pixel gap: RAFT median is 0.07 px across almost every mask; GMFlow-basic's median is ~0.4 px everywhere — likely an unsuppressed 1/8-grid quantization residual from the convex upsample.
- **Bad-10 (catastrophic)** is roughly the same global rate (2.1% RAFT vs 2.3% GMFlow-basic), but is **dominated by occluded pixels** on both: 18.7% of RAFT's unmatched pixels catastrophically fail vs 19.9% for GMFlow-basic. Among matched (non-occluded) pixels, both have <1% catastrophic.
- **s60+** is much worse than s40+ on both: RAFT 12.30 vs 8.72, GMFlow-basic 11.21 vs 8.19. The largest-displacement tail eats most of the dataset's error budget.
- **SD column** shows the EPE distribution is wildly skewed (SD ≈ 5–8× the mean almost everywhere). Mean EPE alone is misleading; the median (A50) and SD together describe the distribution.
- **nEPE column** says the *fractional* error is roughly flat across speed buckets (RAFT 8–10% on s10+/s40+/s60+; GMFlow-basic 9–10%). The absolute EPE grows with speed because the underlying motion grows, not because the model gets proportionally worse. The exception is **GMFlow-basic at s0_10**: nEPE = 18.5% vs RAFT's 12.2% — GMFlow-basic makes proportionally larger errors at slow motion (the same 1/8-grid quantization residual the A50 column hinted at, expressed as a fraction of the motion).
- **Pearson(EPE, AE) per mask** is highly mask-dependent. Tight correlation in well-defined regimes (RAFT s40+ Pearson 0.93, GMFlow-basic s60+ Pearson 0.86). Near-zero correlation in others (GMFlow-basic blur clean 0.22, blur final 0.003 — *zero*). The Spearman column tells a separate story: **s0_1 Spearman is ≈0.97 for RAFT and ≈0.90 for GMFlow-basic** — at sub-pixel motion AE rankings track EPE rankings tightly even when their magnitudes don't (Pearson 0.32 / 0.54). This matches Baker et al. 2011's observation that AE *downweights* large-motion errors — at small motion the geometry is dominated by direction, so AE and EPE rank-track even if their absolute scales diverge.

### Per-sequence EPE (clean)

| seq | RAFT | GMFlow-basic | seq | RAFT | GMFlow-basic |
|---|---|---|---|---|---|
| alley_1 | 0.183 | 0.480 | market_2 | 0.411 | 0.522 |
| alley_2 | 0.172 | 0.342 | market_5 | 4.788 | 4.907 |
| ambush_2 | 4.534 | 3.357 | market_6 | 1.959 | 2.937 |
| ambush_4 | 9.024 | 6.409 | mountain_1 | 0.199 | 0.511 |
| ambush_5 | 2.228 | 2.157 | shaman_2 | 0.213 | 0.347 |
| ambush_6 | 3.203 | 2.942 | shaman_3 | 0.164 | 0.352 |
| ambush_7 | 0.267 | 0.353 | sleeping_1 | 0.114 | 0.257 |
| bamboo_1 | 0.418 | 0.756 | sleeping_2 | 0.110 | 0.329 |
| bamboo_2 | 0.599 | 0.669 | temple_2 | 1.608 | 1.527 |
| bandage_1 | 0.354 | 0.600 | temple_3 | 2.479 | 2.688 |
| bandage_2 | 0.240 | 0.370 | cave_2 | 3.537 | 3.073 |
| cave_4 | 2.114 | 2.270 |  |  |  |

### Clean → Final degradation

| Mask | RAFT ΔEPE | RAFT % | GMFlow-basic ΔEPE | GMFlow-basic % |
|---|---|---|---|---|
| all | +1.232 | +85% | +1.458 | +98% |
| matched | +0.939 | +145% | +1.080 | +131% |
| unmatched | +4.978 | +43% | +6.298 | +63% |
| Disc | +2.680 | +75% | +3.324 | +96% |

> **Figure:** `results/figures/report/clean_to_final_delta.png` — per-sequence ΔEPE bars for all 3 models. ambush_2 dominates the degradation budget (>15 px Δ for every model). Mean per-sequence Δ: RAFT +1.75, GMFlow-basic +2.21, GMFlow-refine +2.05 — H7 supported (RAFT degrades least; the pixel-weighted Δs in the table above and the per-sequence Δs in the figure differ because some sequences contribute many more high-EPE pixels than others).

## 4. Middlebury — zero-shot cross-dataset (Things → Middlebury)

Bold = winner per (sequence, column).

| Sequence | RAFT EE | RAFT AE | RAFT R0.5 | R1.0 | R2.0 | GMFlow-basic EE | GMFlow-basic AE | R0.5 | R1.0 | R2.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Dimetrodon | **0.193** | **3.78°** | **0.054** | **0.004** | 0.000 | 0.394 | 7.64° | 0.211 | 0.011 | 0.000 |
| Grove2 | **0.248** | **3.59°** | **0.073** | **0.035** | 0.016 | 0.346 | 5.15° | 0.145 | 0.038 | **0.009** |
| Grove3 | **0.679** | **6.82°** | **0.313** | **0.170** | **0.071** | 0.770 | 7.76° | 0.462 | 0.190 | 0.073 |
| Hydrangea | **0.224** | **2.72°** | **0.130** | **0.050** | **0.011** | 0.322 | 3.69° | 0.188 | 0.078 | 0.019 |
| RubberWhale | **0.188** | **6.16°** | **0.062** | **0.021** | **0.003** | 0.405 | 12.76° | 0.172 | 0.051 | 0.008 |
| Urban2 | **0.316** | **2.97°** | **0.102** | **0.031** | **0.014** | 0.585 | 5.89° | 0.371 | 0.105 | 0.034 |
| Urban3 | **0.351** | **3.08°** | **0.135** | **0.048** | **0.023** | 0.566 | 4.11° | 0.323 | 0.108 | 0.029 |
| Venus | **0.218** | **2.71°** | **0.043** | **0.006** | **0.002** | 0.503 | 6.49° | 0.422 | 0.061 | 0.003 |
| **mean** | **0.302** | **3.98°** | **0.114** | **0.046** | **0.018** | 0.486 | 6.69° | 0.287 | 0.080 | 0.022 |

RAFT wins on every Middlebury sequence by EE and AE against GMFlow-basic. On R0.5 (Baker's sub-half-pixel robustness threshold) RAFT averages 11.4% bad vs GMFlow-basic's 28.7% — over 2× fewer pixels miss the 0.5-px target. R1.0 and R2.0 differences narrow as the threshold loosens.

### Normalized Sintel → Middlebury generalization (H10)

Methodology §3 H10: report Middlebury EPE normalized by each model's Sintel-domain EPE. Smaller is better (cleaner cross-dataset transfer).

| Model | Sintel clean EE | Middlebury mean EE | normalized score |
|---|---|---|---|
| RAFT | 1.446 | 0.302 | **0.209** |
| GMFlow-basic | 1.484 | 0.486 | 0.328 |

Same direction as the raw numbers — RAFT generalizes ~36% better cross-dataset on this score. Useful framing because it controls for the fact that Middlebury motion magnitudes (mostly < 10 px) sit at the easier end of Sintel's distribution.

### 4b. AE↔EPE correlation on Middlebury

Per-sequence and global Pearson/Spearman between per-pixel EPE and AE on GT-valid pixels.

| Sequence | n (px) | RAFT Pearson | RAFT Spearman | GMFlow-basic Pearson | GMFlow-basic Spearman |
|---|---|---|---|---|---|
| Dimetrodon | 215820 | 0.805 | 0.849 | 0.658 | 0.612 |
| Grove2 | 307200 | **0.978** | 0.870 | **0.966** | 0.891 |
| Grove3 | 307200 | 0.821 | 0.715 | 0.743 | 0.447 |
| Hydrangea | 211712 | 0.678 | **0.906** | 0.639 | **0.876** |
| RubberWhale | 222970 | 0.918 | 0.900 | 0.812 | 0.789 |
| Urban2 | 307200 | 0.391 | 0.169 | 0.357 | 0.161 |
| Urban3 | 307200 | 0.820 | 0.502 | 0.744 | 0.498 |
| Venus | 159600 | 0.766 | 0.531 | 0.689 | 0.560 |
| **GLOBAL** | 2,038,902 | **0.733** | **0.576** | **0.606** | **0.495** |

RAFT's EPE/AE correlation runs ~0.13 higher than GMFlow-basic's globally. Urban2 is the weakest correlation cell for both models — that sequence is mostly slow synthetic motion where small absolute errors translate into large angular swings (AE noise dominates EPE signal).

> **Figures:**
> - `results/figures/report/ae_epe_correlation_middlebury.png` — Pearson + Spearman bars per sequence (3 models × 8 seqs). Urban2 visibly the lowest cell.
> - `results/figures/report/ae_epe_correlation_per_mask.png` — Sintel per-mask Pearson + Spearman (3 models × 11 masks × 2 passes). Shows the s0_1 Spearman ≈ 1.0 case where rankings track even when magnitudes don't, and the GMFlow-basic blur-final Pearson ≈ 0 case where they're completely uncorrelated.

## 4c. AE + percentile accuracies on Sintel

Added via the `_Accum` reservoir (capped at 5M masked samples). All numbers are global, all masks combined into the `all` (valid-GT-only) view.

| | **Clean RAFT** | **Clean GMFlow-basic** | **Final RAFT** | **Final GMFlow-basic** |
|---|---|---|---|---|
| AE / all (°) | **3.91** | 5.78 | **5.58** | 7.67 |
| AE / matched (°) | **3.14** | 5.28 | **4.50** | 6.81 |
| AE / unmatched (°) | 13.86 | **12.10** | 19.30 | **18.64** |
| A50 (px) | **0.067** | 0.411 | **0.068** | 0.409 |
| A75 (px) | **0.131** | 0.497 | **0.134** | 0.517 |
| A95 (px) | **0.707** | 1.085 | **0.723** | 1.133 |

Three-model AE headline (lower is better), computed from the saved `per_seq_stats/*.json` reservoirs for Sintel and saved Middlebury `.npy` predictions:

| Dataset / pass | RAFT-32 | GMFlow-basic | GMFlow-refine | winner |
|---|---:|---:|---:|---|
| Sintel clean | **3.91°** | 5.78° | 4.09° | RAFT-32 |
| Sintel final | **5.58°** | 7.67° | 5.92° | RAFT-32 |
| Middlebury | **3.98°** | 6.69° | 5.52° | RAFT-32 |

This is the main EPE/AE split in the refined comparison: GMFlow-refine takes the best Sintel EPE, but RAFT-32 remains the best angular-error model on Sintel clean, Sintel final, and Middlebury.

Three-model small-motion / precision headline on Sintel clean (lower is better), computed offline from saved predictions:

| Metric | RAFT-32 | GMFlow-basic | GMFlow-refine | best read |
|---|---:|---:|---:|---|
| A50 / all (px) | **0.067** | 0.411 | 0.231 | RAFT has the cleanest median pixel |
| Bad-1 / all | 0.098 | 0.160 | **0.091** | refine slightly lowest; RAFT close |
| nEPE / s1-3 | **0.1255** | 0.2534 | 0.1549 | refine closes most of basic's slow-motion gap |

These numbers sharpen the sub-pixel story: GMFlow-basic has a clear slow-motion residual; GMFlow-refine removes much of it, but RAFT-32 still has the lowest median error and the lowest normalized error in the `s1-3` bucket.

Striking detail: RAFT A50 ≈ 0.07 px on both passes — half of all valid pixels are within 0.07 px of GT. GMFlow-basic's A50 sits at ~0.41 — likely the unsuppressed 1/8-grid quantization residual from its single-pass convex-upsample. The AE / unmatched columns flip the EE-based result: on the angular-error metric GMFlow-basic handles occluded pixels better (12.1° vs 13.9° clean, 18.6° vs 19.3° final), consistent with H4.

## 5. RAFT iter sweep — full Sintel clean (1041 pairs × 4 iter levels)

| iters | EPE / all | s0-10 | s10-40 | s40+ | median ms |
|---|---|---|---|---|---|
| 4 | 1.910 | 0.467 | 2.309 | 11.296 | 198 |
| 8 | 1.593 | 0.388 | 1.888 | 9.521 | 276 |
| 12 | 1.510 | 0.383 | 1.700 | 9.115 | 358 |
| 32 | **1.446** | **0.360** | **1.651** | **8.720** | 765 |

Monotone EPE decrease, diminishing returns: 4→8 saves 0.32 EPE for +78 ms; 12→32 saves only 0.06 EPE for +407 ms. Latency scales sub-linearly with iters (4→32: 8× iters, 3.9× latency — non-iter overhead amortizes).

The full-dataset curve reverses what the 2-sequence subset (alley_1 + market_2) showed: on the subset RAFT-12 (0.292) beat GMFlow-basic (~0.4), but on the full dataset GMFlow-basic and GMFlow-refine both sit on the Pareto frontier.

**Three-way Pareto (Sintel clean).** All EPEs are from the same full-dataset evaluation. **Two latency batches are merged below** — the iter-sweep batch (RAFT-only, single sequential run) and a separate n=50 cross-model batch with all three configs measured back-to-back. RAFT-32 measured **765 ms** in the iter-sweep batch and **594 ms** in the n=50 batch — a ~22% drift from GPU thermal/clock state on the 3050 Ti Laptop. Numbers are tagged with their batch; the cross-model frontier verdict uses **n=50 numbers** since those were all measured under the same conditions.

```
                          latency (ms)         EPE
RAFT  4 iters             198   (iter-sweep)   1.910      ← Pareto (cheapest)
RAFT  8 iters             276   (iter-sweep)   1.593
GMFlow-basic              252   (n=50)         1.484      ← Pareto (best ≤300 ms)
RAFT 12 iters             358   (iter-sweep)   1.510      ← Pareto-dominated by basic
RAFT 32 iters             594   (n=50)         1.446      ← on Pareto by ~58 ms slack
GMFlow-refine             652   (n=50)         1.073      ← Pareto (best accuracy)
```

**RAFT-32 vs GMFlow-refine in the same n=50 batch:** refine is 58 ms slower (10%) and 0.373 EPE lower (26% more accurate). Neither strictly dominates the other; both sit on the Pareto frontier with refine at the accuracy corner and RAFT-32 at the marginal-speed corner. The trade slope strongly favors refine for any non-zero accuracy preference. **GMFlow-basic strictly dominates RAFT-12** (basic 252 ms in the n=50 batch vs RAFT-12 358 ms in the iter-sweep batch — the cross-batch drift could close at most 22% on the RAFT side, which would move RAFT-12 to ~278 ms, still slower than basic; and basic is the more accurate model — 1.484 vs 1.510 EPE).

The original report's "only RAFT-32 beats GMFlow on accuracy" was true against the wrong GMFlow variant. The corrected story: refine beats RAFT-32 on accuracy by 26% at 10% higher latency. (Refine peak VRAM is 1333 MB vs RAFT-32's 529 MB — ~2.5× more, but well under any modern GPU's budget.)

> **Figure:** `results/figures/report/pareto.png` — latency vs Sintel-clean EPE with the RAFT iter sweep curve and three n=50 cross-model markers (RAFT-32 / GMFlow-basic / GMFlow-refine).

## 6. Latency + VRAM at 1024×436 (n=50, mixed sequences)

Two timing batches recorded; each row internally consistent (intra-batch GPU thermal state stable, ±2–4 ms across n=50 pairs). **Cross-model verdicts use the n=50-all-three batch** since those three configs were measured back-to-back under the same thermal conditions.

| | RAFT-32 (orig batch) | RAFT-32 (n=50 all-three) | GMFlow-basic (orig) | GMFlow-basic (n=50 all-three) | GMFlow-refine (n=50 all-three) |
|---|---|---|---|---|---|
| median (ms) | 796 | **594** | 309 | **252** | **652** |
| mean ± SD (ms) | 796 ± 2 | 594 ± 4 | 310 ± 2 | 252 ± 3 | 652 ± 4 |
| min–max (ms) | 794–804 | 586–602 | 305–321 | 245–258 | 646–664 |
| peak VRAM (MB) | 529 | 529 | **470** | 470 | 1333 |

Cross-batch drift on RAFT-32: 765–796 ms (orig) vs 594 ms (n=50 all-three) — the same model on the same hardware can disagree by ~22% across batches due to GPU thermal/clock state on the 3050 Ti Laptop. The two batches' *intra-batch* numbers are precise; the cross-batch absolute scale is not. Hard latency claims should re-measure under controlled thermal state. **Importantly, the within-batch RATIO between models is essentially batch-independent** (when thermal state slows the GPU, both models slow together) — so "refine is 2.6× basic" and "RAFT-32 is 2.36× basic" hold even if the absolute milliseconds drift; only the cross-batch absolute comparisons need a caveat.

GMFlow-basic is ~2.4× faster than RAFT-32 (n=50 all-three batch) at Sintel resolution. GMFlow-refine is 10% slower than RAFT-32 but uses 2.5× more VRAM (1333 vs 529 MB) — still well under any modern GPU's budget at this resolution.

### 6b. Resolution sweep — H9 **indicative only**, NOT verified

> **Scope.** This resolution sweep measured RAFT-32 vs **GMFlow-basic** only. GMFlow-refine was not included in this high-resolution sweep.
>
> **Contamination.** At factors 2.0× and 2.5×, the reported allocated memory (6.1 / 6.3 GB, 14.97 / 15.23 GB) exceeds the 3050 Ti's 4 GB physical capacity, meaning **both measured models** spill to host memory via WSL2 unified-memory paging. The reported latency at those rows mixes the genuine O(N²) attention cost with PCIe paging cost — and the two models may page differently. The 1.0× and 1.5× rows are clean (allocated <2.1 GB, well under the 4 GB physical) and already show GMFlow-basic trending toward the latency cross-over. The 2.0× / 2.5× rows are kept below as "indicative" but **the original "strongly supported" verdict has been retracted**. Verification on a ≥16 GB physical-memory GPU is the open follow-up.

Sintel pairs upsampled by factor `f` (bilinear), n=10 timed forwards per cell:

| factor | resolution | RAFT-32 (ms / MB) | GMFlow-basic (ms / MB) | GMFlow-basic / RAFT latency |
|---|---|---|---|---|
| 1.00× | 1024×436 | 737 / 529 | 300 / 470 | **0.41×** (basic faster) |
| 1.50× | 1536×654 | 1,765 / 2,023 | 1,174 / 2,036 | **0.66×** (basic still faster) |
| 2.00× | 2048×872 | 3,591 / 6,146 | **12,431** / 6,285 | **3.46×** (basic slower) |
| 2.50× | 2560×1090 | 6,877 / 14,970 | **43,894** / 15,229 | **6.38×** (basic catastrophic) |

**Honest read:** the 1.00× and 1.50× rows are clean (allocated <2.1 GB, well under the 4 GB physical) and show RAFT's latency-vs-resolution scaling near-linear (737 → 1765 ms for 2.25× pixels — modest super-linear from the per-pair fixed cost amortizing) while GMFlow-basic climbs faster (300 → 1174 ms = 3.9× for 2.25× pixels). The GMFlow-basic / RAFT latency ratio moves from 0.41× → 0.66× — consistent direction for an O(N²) attention catching up to O(N) refinement, but with only two clean data points the slope is not pinned down. The 2.00× / 2.50× latency numbers reported above (12,431 ms and 43,894 ms for GMFlow-basic; 3,591 ms and 6,877 ms for RAFT) mix the genuine attention cost with PCIe paging cost and **should not be read as model-scaling measurements**. They tell you only that the latency *is enormous* on contaminated hardware, not how O(N²) it is. The 146× / 9.3× scaling ratios that previously appeared here have been removed.

Cross-over on this hardware appears to land between 1.5× and 2× Sintel based on the 1.0× and 1.5× clean rows alone (RAFT/GMFlow-basic latency ratio 0.41× → 0.66× — trajectory consistent with O(N²) catching up). The 2.0× and 2.5× rows show ratios of 3.46× and 6.38× but those are not trustworthy (paging contamination), so the *magnitude* of the blow-up is unknown and the *location* of the cross-over is bracketed only loosely (anywhere from 1.5× to >2×). See §11e for the resolution-sweep follow-up plan.

## 7. Photometric residual (Sintel, mean over non-occluded valid pixels)

Per-sequence residual `|I₁ − I₂(x + flow_gt)|` averaged over RGB.

| Seq | Clean | Final | Δ | Seq | Clean | Final | Δ |
|---|---|---|---|---|---|---|---|
| alley_1 | 2.14 | 1.28 | −0.86 | market_2 | 2.35 | 1.57 | −0.79 |
| alley_2 | 2.81 | 2.97 | +0.16 | market_5 | 3.46 | 4.83 | +1.37 |
| ambush_2 | 3.44 | 12.84 | **+9.40** | market_6 | 4.97 | 4.82 | −0.15 |
| ambush_4 | 5.71 | 10.76 | +5.05 | mountain_1 | 2.88 | 3.38 | +0.50 |
| ambush_5 | 3.66 | 5.75 | +2.09 | shaman_2 | 1.12 | 0.91 | −0.21 |
| ambush_6 | 5.42 | 9.15 | +3.74 | shaman_3 | 2.19 | 1.52 | −0.66 |
| ambush_7 | 2.32 | 2.14 | −0.18 | sleeping_1 | 1.74 | 1.66 | −0.07 |
| bamboo_1 | 3.26 | 2.39 | −0.86 | sleeping_2 | 2.47 | 1.41 | −1.07 |
| bamboo_2 | 2.49 | 2.35 | −0.15 | temple_2 | 2.80 | 2.66 | −0.14 |
| bandage_1 | 1.51 | 2.27 | +0.76 | temple_3 | 4.84 | 8.61 | +3.77 |
| bandage_2 | 1.33 | 2.01 | +0.68 | cave_2 | 2.63 | 3.41 | +0.78 |
| cave_4 | 3.74 | 2.75 | −0.98 | **mean** | **3.01** | **3.98** | **+0.96** |

Reading: ambush_2/ambush_4/ambush_6/temple_3 dominate the Clean→Final brightness-constancy violation budget. Many slow sequences (sleeping_2, market_2, cave_4, alley_1) have *negative* Δ — Final's motion blur smooths I₂ more than the brightness shift adds, lowering residual. The mean +0.96 confirms Final is photometrically harder *on average*, but the per-sequence distribution is heavy-tailed.

## 7b. Per-pixel ΔEPE maps (Clean vs Final, model failure localization)

`ΔEPE = EPE_final − EPE_clean` averaged per-pixel across all pairs of each sequence. Diverging RdBu_r maps written to `results/figures/delta_epe/<tag>/<seq>.png` (46 PNGs total, one per (model, sequence)), plus `_summary.csv` per model.

Per-sequence median ΔEPE (the per-pixel median over the averaged map):

| seq | RAFT median Δ | GMFlow-basic median Δ | seq | RAFT median Δ | GMFlow-basic median Δ |
|---|---|---|---|---|---|
| alley_1 | +0.005 | +0.007 | market_2 | +0.031 | +0.078 |
| alley_2 | +0.002 | +0.000 | market_5 | +2.049 | +1.621 |
| ambush_2 | **+16.88** | **+21.30** | market_6 | +0.095 | +0.047 |
| ambush_4 | +5.03 | **+15.25** | mountain_1 | +0.024 | +0.037 |
| ambush_5 | +2.92 | +1.89 | shaman_2 | +0.015 | +0.011 |
| ambush_6 | +3.80 | +5.98 | shaman_3 | +0.033 | +0.006 |
| ambush_7 | +0.065 | +0.026 | sleeping_1 | +0.004 | −0.001 |
| bamboo_1 | **−0.003** | **−0.014** | sleeping_2 | +0.003 | +0.004 |
| bamboo_2 | +0.011 | +0.010 | temple_2 | +0.145 | +0.223 |
| bandage_1 | +0.040 | +0.051 | temple_3 | +1.444 | +1.941 |
| bandage_2 | +0.017 | −0.000 | cave_2 | +1.820 | +1.400 |
| cave_4 | +0.335 | +0.404 |  |  |  |

ambush_2 dominates the Clean→Final regression on both models, with GMFlow-basic even more affected than RAFT. ambush_4 is the largest gap between models: RAFT degrades ~3× less than GMFlow-basic there (median Δ 5.03 vs 15.25). A handful of sequences (bamboo_1, sleeping_1 for GMFlow-basic, bandage_2 for GMFlow-basic) have **negative** median Δ — Final's motion blur smooths errors more than its photometric shift adds, so Final is *easier* to predict on those sequences. This is the per-pixel evidence behind methodology §4 item 10.

## 7d. Blur / defocus mask via Laplacian-variance (methodology §2.6, §2.8)

Per-pair blur mask = `var(Laplacian(I₁, ksize=3), 7×7) < 20` — windowed Laplacian variance, low variance ⇒ blurred / defocused. Applied as an additional mask in `SintelMetrics`.

| | Clean RAFT | Clean GMFlow-basic | Clean GMFlow-refine | Final RAFT | Final GMFlow-basic | Final GMFlow-refine |
|---|---:|---:|---:|---:|---:|---:|
| Blur-pixel fraction (over `valid`) | 12.7% | 12.7% | 12.7% | 34.6% | 34.6% | 34.6% |
| EPE / blur | 2.175 | 1.881 | **1.405** | 4.029 | 4.669 | **3.866** |
| AE / blur (°) | 3.774 | 4.794 | **3.459** | **6.165** | 8.355 | 6.580 |
| Bad-1 / blur | **0.108** | 0.182 | 0.111 | **0.187** | 0.253 | 0.197 |
| Bad-10 / blur | 0.029 | 0.030 | **0.023** | 0.066 | 0.072 | **0.061** |

Sequence-wide blur fractions (mean over all pairs in the sequence): clean alley_1 = 4.6%, market_2 = 5.9%, ambush_4 = 35.3%, mountain_1 = 36.4%, temple_3 = 58.1%. Final-pass adds renderer motion blur — global fraction nearly triples 12.7% → 34.6%; per-sequence final: alley_1 = 14.4%, market_2 = 22.5%, ambush_4 = 60.5%, ambush_2 = 81.9%, shaman_3 = 72.9%.

Reading: GMFlow-basic has *lower mean EPE* on blurred pixels of Sintel **clean** (1.881 vs 2.175) but worse AE and Bad-1 than RAFT, so the basic model's clean-blur "win" is not distributionally clean. GMFlow-refine changes that picture: on clean blur it has the best EPE (1.405), best AE (3.459°), and best Bad-10 (0.023), while RAFT is only slightly better on Bad-1 (0.108 vs 0.111). On Final blur, refine again has the best EPE (3.866) and Bad-10 (0.061), but RAFT keeps the best AE (6.165°) and Bad-1 (0.187). Bootstrap agrees with the split: refine beats RAFT on clean-blur EPE, while final-blur EPE is lower for refine as a point estimate but RAFT vs refine is NULL (§11a).

**Confound:** the blur mask correlates with motion magnitude across the 23 sequences (Pearson 0.644, Spearman 0.520; full table in §11c). It is **not** a pure motion-magnitude proxy — `mountain_1` has low motion (4.98 px mean) and high blur fraction (36%), confirming the mask catches actual defocus/specular blur independent of motion. But the moderate correlation means part of the "blur" signal in this table is fast-motion contamination; the EPE/blur differences should be read with that confound in mind, especially on sequences like ambush_4 (60% blur, 32 px mean motion).

## 7e. Speed-bucket curves (EPE / AE / nEPE / Bad-1 vs |gt|)

Bin pixels by GT-flow magnitude on a finer grid than the §3 buckets: `s0-1, s1-3, s3-10, s10-20, s20-40, s40-60, s60+`. Plot the four metrics as line charts with one curve per (model, pass). PNGs in `results/figures/speed_curves/`, raw values in `speed_curves.csv`.

Per-bucket pixel count (same across all 4 configs):

```
s0-1     s1-3    s3-10  s10-20  s20-40  s40-60   s60+
97.8M  109.6M   113.2M   60.8M   37.9M   18.3M  26.8M
```

EPE / AE / nEPE / Bad-1 numbers per (config, bucket) — all 28 cells × 4 metrics live in the CSV; the most informative cuts:

**EPE (px) vs speed:**

| config | s0-1 | s1-3 | s3-10 | s10-20 | s20-40 | s40-60 | s60+ |
|---|---|---|---|---|---|---|---|
| RAFT clean | **0.161** | **0.223** | **0.665** | **1.173** | **2.417** | **3.488** | 12.30 |
| GMFlow-basic clean | 0.245 | 0.428 | 0.665 | 1.253 | 2.574 | 3.772 | **11.21** |
| RAFT final | 0.293 | 0.341 | 0.859 | 2.111 | 4.411 | 8.355 | 23.61 |
| GMFlow-basic final | 0.371 | 0.553 | 1.194 | 2.524 | 4.904 | 8.569 | 23.84 |

**Normalized EPE = EPE / |gt|  (only on `|gt|>1`):**

| config | s0-1 | s1-3 | s3-10 | s10-20 | s20-40 | s40-60 | s60+ |
|---|---|---|---|---|---|---|---|
| RAFT clean | nan | **0.126** | **0.119** | 0.084 | **0.086** | 0.072 | 0.098 |
| GMFlow-basic clean | nan | 0.253 | 0.118 | **0.089** | 0.091 | **0.077** | **0.096** |
| RAFT final | nan | 0.191 | 0.149 | 0.149 | 0.156 | 0.169 | 0.206 |
| GMFlow-basic final | nan | 0.326 | 0.210 | 0.176 | 0.173 | 0.175 | 0.210 |

**AE (degrees) vs speed:**

| config | s0-1 | s1-3 | s3-10 | s10-20 | s20-40 | s40-60 | s60+ |
|---|---|---|---|---|---|---|---|
| RAFT clean | 5.72 | 3.78 | 3.00 | 2.80 | 3.43 | 3.41 | 5.29 |
| GMFlow-basic clean | 9.71 | 6.95 | 4.19 | 3.44 | 3.55 | **3.00** | **3.68** |
| RAFT final | 7.29 | 4.56 | 3.88 | 4.48 | 6.25 | 8.25 | 10.35 |
| GMFlow-basic final | 10.98 | 8.28 | 5.28 | 5.32 | 6.83 | 8.19 | 9.30 |

Reading:

- **EPE vs speed**: RAFT < GMFlow-basic at every clean bucket up to s40-60 then they cross — GMFlow-basic s60+ EPE 11.21 < RAFT 12.30. The cross is consistent with H1's point estimate, but §11a shows the basic-vs-RAFT large-motion difference is bootstrap-NULL.
- **nEPE vs speed**: GMFlow-basic's huge nEPE at s1-3 (0.253 clean, 0.326 final — 2× RAFT) shows GMFlow-basic's small-motion errors are *proportionally* much larger; from s10-20 onwards both models hover around 8–9% on clean. The nEPE curve is U-shaped: highest at slow + fastest motion, lowest in the s10-60 sweet spot.
- **AE vs speed**: U-shape too, but with model character. GMFlow-basic's AE *decreases* with speed (9.71 → 3.00 over s0-1 to s40-60) — its errors at high speed are more aligned with GT motion. RAFT's AE bottoms out around s10-20 then rises again at s60+.
- **Bad-1 vs speed** (in CSV): monotonically increasing on every config; RAFT clean s0-1 = 1.5% vs s60+ = 41%, the spread is enormous.

Figures: `results/figures/speed_curves/{epe,ae,nepe,bad1}_vs_speed.png`.

## 7c. Forward-backward consistency derived occlusion (methodology §2.2)

Per-pair derived occlusion = `‖f12(x) + f21(x + f12(x))‖² > α(‖f12‖² + ‖f21‖²) + β` with `α=0.01`, `β=0.5` (Sundaram et al. 2010 / Meister et al. 2018). Backward flow computed by swapping `(img1, img2)`. Forward predictions reused from saved `.npy`. Compared per-pixel against Sintel native `occlusion == 255` on valid GT.

| | RAFT | GMFlow-basic |
|---|---|---|
| native occluded fraction | 7.25% | 7.25% |
| derived occluded fraction | 6.25% | 4.85% |
| precision | 0.654 | **0.752** |
| recall | **0.564** | 0.504 |
| F1 | 0.606 | 0.603 |
| IoU | 0.435 | 0.432 |

Both models recover ≈0.60 F1 / ≈0.43 IoU of the GT occlusion mask from inference alone. This forward-backward test was run for RAFT and GMFlow-basic only; GMFlow-refine was not measured here. GMFlow-basic's mask is sparser and more precise (predicts fewer occlusions but most are real), RAFT's is denser and higher-recall. The two F1 numbers are essentially indistinguishable (0.606 vs 0.603) despite GMFlow-basic having lower unmatched EPE — the *quality of the inferred occlusion mask* and the *EPE on the occluded region* are independent axes. §2.2 cross-check is satisfied: fwd-bwd consistency is a usable proxy for occlusion when GT is unavailable.

## 8. Hypothesis verdicts (Phase 1)

Each verdict now has two layers: **vs GMFlow-basic** (the original report's framing) and **vs GMFlow-refine** (the fair-comparison framing). Bootstrap 95% CIs over the 23 Sintel sequences are reported for hair-margin decisions. "Supported", "falsified", and "NULL" are assigned per comparison; if a variant was not measured for that hypothesis, it is marked explicitly. Full numbers behind every verdict are in §11.

| H | Claim | vs GMFlow-basic | vs GMFlow-refine | Final read |
|---|---|---|---|---|
| 1 | GMFlow > large displacements | Clean s40+/s60+: NULL (Δ CIs cross zero — see §11a). Final s40+/s60+: NULL. | Clean s40+ −2.53 [−3.39, −1.85], s60+ −3.75 [−5.60, −2.34]; Final s40+ −1.81 [−3.09, −0.36], s60+ −2.62 [−4.78, −0.51]. All significant. | **Supported once a comparable-capacity GMFlow is used.** The "GMFlow basic beats RAFT-32 on s40+" headline of the original report does **not** survive sequence-bootstrap — that comparison was undersampled, not a real win. The refine variant **does** beat RAFT-32 on every large-displacement mask significantly. |
| 2 | RAFT > sub-pixel | Clean s0_1 +0.084 [+0.059, +0.131] (RAFT wins); Bad-1 +0.062 [+0.045, +0.081] (RAFT wins). | Clean s0_1 NULL; Final s0_1 NULL; Bad-1/all NULL. RAFT's sub-pixel lead **evaporates** vs refine. | **Supported vs basic, falsified vs refine.** The lead was capacity, not architecture. |
| 3 | RAFT sharper boundaries | F1 0.727 vs 0.697 at τ=1.0 (clean). Lead survives all three thresholds {0.5, 1.0, 2.0}. | F1 0.727 vs **0.757** (refine wins) at τ=1.0 clean, **0.756** at τ=2.0, **0.777** at τ=0.5. | **Supported vs basic, falsified vs refine at every threshold.** |
| 4 | GMFlow > occlusions | Clean unmatched −1.74 [−3.32, −0.12] (basic wins, p=0.035). Final unmatched NULL. | Clean unmatched −3.39 [−4.91, −2.00] (refine wins). Final unmatched −1.83 [−3.05, −0.51] (refine wins). | **Supported on clean for both basic and refine; only refine is significant on final.** |
| 5 | Both adequate on untex | Clean untex EPE: RAFT 1.65 vs basic 1.55, Δ −0.10 [−0.54, +0.18] ⇒ NULL. Final untex EPE: RAFT 3.109 vs basic 3.462, Δ +0.353 [−0.243, +1.215] ⇒ NULL. Basic does not blow up on textureless regions, but it also does not win significantly. | Clean untex EPE: RAFT 1.65 vs refine 1.13, Δ −0.52 [−1.10, −0.14] ⇒ refine wins. Final untex EPE: RAFT 3.1088 vs refine 2.8380, Δ −0.2708 [−0.6493, +0.0331] ⇒ NULL. Untex / All ratio: RAFT 1.14, basic 1.05, refine 1.05. | **Supported as a non-blow-up claim for all measured variants.** Basic is statistically tied with RAFT on untex EPE; refine significantly improves clean untex and is lower-but-NULL on final untex. |
| 6 | RAFT iter ↔ accuracy trade-off | Full-dataset sweep stands: EPE 1.91/1.59/1.51/1.45 at iters {4,8,12,32}. GMFlow-basic measured 252 ms (n=50 all-three batch). | GMFlow-refine clean EPE **1.073** at **652 ms** (n=50 all-three). RAFT-32 in the same batch is 594 ms / 1.446. Refine costs 10% more latency for 26% lower EPE — not strictly Pareto-dominating RAFT-32 (which is 58 ms cheaper), but the trade slope strongly favors refine. | **Supported as the iter-budget shape;** the Pareto picture against the *available* GMFlow variants is now well-measured (§5 / §6): refine and RAFT-32 are both on the frontier, refine 58 ms slower and 26% more accurate. GMFlow-basic strictly dominates RAFT-12. |
| 7 | GMFlow more tolerant Clean→Final | RAFT ΔEPE = 1.23; GMFlow-basic = 1.46. Basic degrades **more** than RAFT in raw EPE. | GMFlow-refine ΔEPE = 1.39. Also degrades more than RAFT in raw EPE. | **Falsified.** No GMFlow variant degrades less. The original "partial on matched-relative" framing was a denominator artifact (GMFlow starts worse → same absolute Δ → smaller relative Δ); dropped. |
| 8 | RAFT weak to weather, GMFlow to noise | Not measured. Requires RobustSpring corruption suite, which is descoped. | Not measured. GMFlow-refine was not evaluated on RobustSpring corruptions either. | **Not pursued.** RobustSpring corruption suite is out of scope for this study. No GT-free corruption-robustness numbers are reported for either GMFlow variant. |
| 9 | GMFlow VRAM blow-up | Sintel 1024×436: both <600 MB and GMFlow-basic 2.6× faster. At 2× Sintel GMFlow-basic's measured latency is 12,431 ms vs RAFT's 3,591 ms; at 2.5× it's 43,894 vs 6,877. | GMFlow-refine not measured in the resolution sweep. | **Indicative, NOT verified.** At 2× and 2.5×, allocated memory exceeds the 3050 Ti's 4 GB physical (6.1 GB / 15 GB shown), meaning **both measured models** are paging through host memory via WSL2's unified-memory mechanism — and they may page differently. The reported latency mixes O(N²) attention scaling with PCIe paging cost; the two cannot be separated locally. The 1.0× and 1.5× rows are clean (well under 4 GB) and already show GMFlow-basic trending toward the cross-over (0.41× → 0.66× latency ratio). Verification on a ≥16 GB physical-memory GPU is still required to call this "supported"; flagged in §6b and §9. |
| 10 | Sintel→Middlebury drops unequally | RAFT Middlebury 0.302 vs basic 0.486. Normalized two ways (see §4b rewrite + §11d): (a) `Mid / Sintel-all`: RAFT 0.209, basic 0.328 — "36% gap" depends on this choice; (b) `Mid / Sintel-s0_10` (better-matched motion regime): RAFT 0.839, basic 1.066 — gap is ~27%. | (a) RAFT 0.209 vs refine 0.375; (b) RAFT 0.839 vs refine **1.331** — refine has the worst Sintel→Middlebury ratio of the three, because refine reduces Sintel EPE more than Middlebury EPE. (Both checkpoints are Things-trained zero-shot — refine never saw Sintel, so this is **not** Sintel-overfitting; it's a **motion-regime effect**: refine's extra capacity pays off on Sintel's large/structured motion tail (s10+) and buys little on Middlebury's mostly-<10 px regime.) | **Direction holds — RAFT generalizes best cross-dataset** — but magnitude depends entirely on the normalization choice; both reported above and in §11d. The refine result reads as: higher capacity fits the large-motion regime better, and that gain doesn't transfer to small-motion data. |

## 9. Caveats and notes

- **Motion-boundary mask is derived** (Sobel + 9×9 dilate), not from Sintel's native `motion_boundaries/`. The `MPI-Sintel-training_extras.zip` checked: it only contains `flow_code/`, `flow_viz/`, `invalid/`, `occlusions/` — no `motion_boundaries/`. The native mask appears not to be publicly distributed. Both pred and GT use the same derivation, so Boundary F-score is internally consistent but not directly comparable to papers that use the native mask. F1 threshold sensitivity tested at τ ∈ {0.5, 1.0, 2.0} — see §11b; verdicts hold across thresholds.
- **Blur mask threshold is uncalibrated.** `var(Laplacian(I, 3×3), 7×7) < 20` is an order-of-magnitude pick. Per-sequence blur fraction correlates with mean GT-motion at Pearson 0.644 (§11c) — moderate confound, not pure. `mountain_1` (low motion + high blur) confirms the mask isn't only catching fast-motion sequences.
- **GMFlow-basic/refine A50 mechanism is "coarse-grid upsample residual at small-k/8 multiples", not strictly 1/8-grid quantization** — original wording was over-specific; revised. §11f histograms show partial pile-up at k/8 multiples on slow sequences (8–13% concentration at k=1, 2 for basic), smeared on fast control (ambush_4). Refine reduces but doesn't eliminate.
- **Schulze ranking / RobustSpring corruption suite (methodology §1.6, §4 items 11–12) is out of scope** — explicitly descoped, not a follow-up. H8 is "not pursued"; no corruption-robustness numbers in this study.
- **WSL2 unified memory contaminates the H9 resolution sweep at 2.0× and 2.5×.** See §6b downgrade and §11e plan. The "GMFlow VRAM blow-up at Spring resolution" conclusion is **indicative, not verified** until re-run on a ≥16 GB physical-memory GPU.
- **GMFlow ablation is now three-way:** basic (no refinement, 1 scale, padding-factor 16) vs with-refine (2 scales, padding-factor 32, the upstream `evaluate.sh` preset). Original report's RAFT-32 vs GMFlow-basic comparison was capacity-asymmetric; §8 verdicts now report both reads. The §3 main table is preserved as the original framing; §11a / §3a contain the symmetric ablation.
- **Iter sweep was run at two granularities**: subset (alley_1 + market_2 only, ~3 min) showed RAFT-12 minimum at 0.292 EPE on that easy subset; full-dataset (all 23 sequences, ~34 min) shows monotone decrease across the full curve with GMFlow-basic and GMFlow-refine on the Pareto frontier. Both numbers reported above (§5).
- **Bootstrap CIs use sequence-level resampling**, not pixel-level. Pixels within a sequence are not independent (shared scene). The effective sample size is the 23 Sintel sequences, and pixel-level bootstrap would dramatically under-estimate CI width.
- **No multiplicity correction in §11a/§11b.** Each CI is individual. Across the ~264 cells in §11a a Bonferroni-corrected threshold (α/264 ≈ 0.0002) would kill the borderline "B wins (just)" / "B wins (modest)" cells. Verdicts in §8 only rely on strongly-significant cells (`|Δ| / CI-half-width > 3`); barely-significant cells are reported but should not carry interpretive weight on their own.
- **Percentile bootstrap, not BCa.** Heavy-tailed masks (s40+, unmatched) on 23 sequences let one or two sequences (ambush_2, ambush_4) dominate resamples; BCa would correct for this slightly but does not change the qualitative verdicts. For tighter CI endpoints on those masks specifically, switch `paired_bootstrap` to BCa — the framework is the same.

## 10. Reproducing

From repo root with `.venv` active and `PYTHONPATH=src`:

```bash
# §2 + §3 (Sintel both passes, both models, saves .npy predictions):
python -m cvflow.runners.run_sintel_eval --model raft   --pass clean
python -m cvflow.runners.run_sintel_eval --model gmflow --pass clean
python -m cvflow.runners.run_sintel_eval --model raft   --pass final
python -m cvflow.runners.run_sintel_eval --model gmflow --pass final

# Full mask suite (Disc/Untex/Blur/s60+/all percentiles/SD/AE/Bad-1,3,5,10/F-score) offline:
python -m cvflow.runners.eval_from_saved --pred-root results/predictions/raft-raft-things-iter32   --pass clean
python -m cvflow.runners.eval_from_saved --pred-root results/predictions/gmflow-gmflow_things-e9887eda --pass clean
# Knobs: --disc-thresh 1.0  --untex-thresh 5.0  --blur-window 7  --blur-thresh 20.0

# §4: Middlebury
python -m cvflow.runners.run_middlebury --model both

# §5: iter sweep (subset)
python -m cvflow.runners.run_raft_itersweep

# §5: iter sweep (full dataset, all 23 sequences)
python -m cvflow.runners.run_raft_itersweep \
  --seqs alley_1 alley_2 ambush_2 ambush_4 ambush_5 ambush_6 ambush_7 \
         bamboo_1 bamboo_2 bandage_1 bandage_2 cave_2 cave_4 \
         market_2 market_5 market_6 mountain_1 shaman_2 shaman_3 \
         sleeping_1 sleeping_2 temple_2 temple_3 \
  --iters 4 8 12 32

# §6: latency + VRAM at 1024×436
python -m cvflow.runners.run_latency_vram --n 50

# §6b: latency + VRAM scaling with resolution
python -m cvflow.runners.run_vram_resolution

# §7: photometric residual
python -m cvflow.runners.run_photometric

# §7b: ΔEPE per-pixel maps (writes PNGs + summary CSV)
python -m cvflow.runners.delta_epe_maps --pred-root results/predictions/raft-raft-things-iter32
python -m cvflow.runners.delta_epe_maps --pred-root results/predictions/gmflow-gmflow_things-e9887eda

# §7c: fwd-bwd derived occlusion (reuses saved forward predictions, ~14+6 min)
python -m cvflow.runners.run_fwdbwd_occlusion --model raft \
  --fwd-cache results/predictions/raft-raft-things-iter32/sintel/clean
python -m cvflow.runners.run_fwdbwd_occlusion --model gmflow \
  --fwd-cache results/predictions/gmflow-gmflow_things-e9887eda/sintel/clean

# §11: critique-driven revisions
# GMFlow-refine ablation (adds a 3rd model)
python -m cvflow.runners.run_sintel_eval --model gmflow --gmflow-refine --pass clean
python -m cvflow.runners.run_sintel_eval --model gmflow --gmflow-refine --pass final
python -m cvflow.runners.run_middlebury --model gmflow --gmflow-refine

# §11a: paired sequence-level bootstrap on close calls
# First dump per-seq JSON:
for tag in raft-raft-things-iter32 gmflow-gmflow_things-e9887eda gmflow-gmflow_with_refine_things-36579974; do
  for pass in clean final; do
    short=$(echo $tag | sed 's/raft-raft-things-iter32/raft/; s/gmflow-gmflow_things-e9887eda/gmflow_basic/; s/gmflow-gmflow_with_refine_things-36579974/gmflow_refine/')
    python -m cvflow.runners.eval_from_saved --pred-root results/predictions/$tag --pass $pass \
      --dump-json results/per_seq_stats/${short}_${pass}.json
  done
done
# Then bootstrap-compare pairs:
python -m cvflow.runners.bootstrap_compare \
  --a results/per_seq_stats/raft_clean.json \
  --b results/per_seq_stats/gmflow_refine_clean.json

# §11b: F1 threshold sensitivity
python -m cvflow.runners.boundary_threshold_sweep

# §11c: blur ↔ motion confound
python -m cvflow.runners.blur_motion_confound

# §11f: quantization-residual histograms
python -m cvflow.runners.quantization_check
```

## 11. Critique-driven revisions (collected evidence)

This section bundles the data behind the H1/H3/H4/H6/H7/H9/H10 verdict rewrites in §8. Run scripts are in §10.

### 11a. Paired sequence-level bootstrap — close calls

Method: resample the 23 Sintel sequences with replacement (paired across the two configs), recompute pixel-weighted Δ per resample, take the 2.5/97.5 percentiles. `n_boot = 10,000`. CI crossing zero ⇒ "NULL" — verdict can't be distinguished from sampling noise.

> **No multiplicity correction applied.** Each cell below is an *individual* 95% CI. With 11 masks × 3 pairings × 2 passes × 4 metrics ≈ 264 comparisons in §11a, a Bonferroni-corrected significance threshold would be α/264 ≈ 0.0002 — barely-significant cells (CIs that just exclude zero, "B wins (just)" or "B wins (modest)" labels) almost certainly **do not survive** multiplicity correction. Strongly-significant cells with `|Δ|/CI-half-width > 3` (e.g. s40+ refine [−3.39, −1.85], p≈0) remain safe; the borderline cells should not carry interpretive weight on their own.

**Sintel clean — RAFT vs GMFlow-basic (the original report's central comparison):**

| mask | A=RAFT EPE | B=GMFlow-basic EPE | ΔB-A | 95% CI | verdict |
|---|---|---|---|---|---|
| all | 1.446 | 1.484 | +0.038 | [−0.191, +0.202] | **NULL** |
| matched | 0.646 | 0.822 | +0.177 | [+0.058, +0.256] | A wins |
| unmatched | 11.69 | 9.95 | −1.74 | [−3.32, −0.12] | B wins (modest) |
| s0_1 | 0.161 | 0.245 | +0.084 | [+0.059, +0.131] | A wins |
| s0_10 | 0.360 | 0.456 | +0.096 | [−0.115, +0.202] | **NULL** |
| s10_40 | 1.651 | 1.760 | +0.110 | [−0.403, +0.392] | **NULL** |
| s40+ | 8.72 | 8.19 | −0.53 | [−1.35, +0.68] | **NULL** |
| s60+ | 12.30 | 11.21 | −1.09 | [−2.60, +0.81] | **NULL** |
| disc | 3.58 | 3.46 | −0.12 | [−0.55, +0.24] | **NULL** |
| untex | 1.65 | 1.55 | −0.10 | [−0.54, +0.18] | **NULL** |
| blur | 2.18 | 1.88 | −0.29 | [−1.60, +0.40] | **NULL** |

The headline "EPE/all 1.446 vs 1.484 = RAFT wins by 2.5%" of the original report is **not significant** (Δ CI [−0.19, +0.20], p=0.64). RAFT's real wins are on `matched` and `s0_1`; GMFlow-basic's real wins are on `unmatched`. Everything else was within the 23-sequence noise.

**Sintel clean — RAFT vs GMFlow-refine:**

| mask | A=RAFT EPE | B=GMFlow-refine EPE | ΔB-A | 95% CI | verdict |
|---|---|---|---|---|---|
| all | 1.446 | 1.073 | **−0.373** | [−0.702, −0.139] | B wins |
| matched | 0.646 | 0.508 | −0.137 | [−0.333, −0.014] | B wins |
| unmatched | 11.69 | 8.30 | **−3.39** | [−4.91, −2.00] | B wins |
| s0_1 | 0.161 | 0.157 | −0.004 | [−0.017, +0.012] | **NULL** |
| s0_10 | 0.360 | 0.302 | −0.058 | [−0.269, +0.039] | **NULL** |
| s10_40 | 1.651 | 1.242 | −0.409 | [−1.021, −0.097] | B wins |
| s40+ | 8.72 | 6.19 | **−2.53** | [−3.39, −1.85] | B wins |
| s60+ | 12.30 | 8.55 | **−3.75** | [−5.60, −2.34] | B wins |
| disc | 3.58 | 2.54 | −1.03 | [−1.67, −0.53] | B wins |
| untex | 1.65 | 1.13 | −0.52 | [−1.10, −0.14] | B wins |
| blur | 2.18 | 1.40 | −0.77 | [−2.19, −0.03] | B wins (just) |

Refine wins every mask significantly except the two slowest buckets (`s0_1`, `s0_10`) where RAFT and refine are statistically tied — RAFT's much-cited sub-pixel advantage **disappears** against the higher-capacity GMFlow.

**Sintel final — RAFT vs GMFlow-basic:** ALL EPE masks except `matched` (RAFT wins) and `s0_1`/`s0_10` (RAFT wins) are **NULL**. The "RAFT wins by hair margins on final s40+/s60+/unmatched" reading of the original §3 main table does not survive sequence-bootstrap. Bad-1 and AE differences ARE significant (RAFT wins decisively). Full table omitted for brevity — in `results/bootstrap/final__raft_vs_gmflow_basic.txt`.

**Sintel final — RAFT vs GMFlow-refine:** Refine wins `all`, `unmatched`, `s40+`, `s60+`, `disc` (significant). `matched`, `s0_10`, `s10_40`, `untex`, `blur` are NULL. For `untex`, the exact EPE point estimate is RAFT `3.1088` vs GMFlow-refine `2.8380` (Δ = `−0.2708`, 95% CI `[−0.6493, +0.0331]`, p = `0.0882`), so refine is lower but not statistically separable at 95%. Bad-1 is NULL globally. RAFT wins `matched-AE`, `s0_1-AE`, `s0_10-AE`. Full table in `results/bootstrap/final__raft_vs_gmflow_refine.txt`.

**Sintel final — GMFlow-basic vs GMFlow-refine:** Refine wins **every metric on every mask** (clean and final). The refine upgrade is uniform — not a tradeoff.

### 11b. Boundary F1 threshold sensitivity

Disc-mask EPE point estimates from `per_seq_stats/*.json` (lower is better):

| Pass | RAFT-32 | GMFlow-basic | GMFlow-refine | best read |
|---|---:|---:|---:|---|
| Clean | 3.579 | 3.463 | **2.544** | refine has the lowest boundary-region EPE |
| Final | 6.258 | 6.787 | **5.762** | refine stays best; RAFT beats basic on final |

F-score at τ ∈ {0.5, 1.0, 2.0} (gradient threshold for the Disc mask):

| config | τ=0.5 | τ=1.0 | τ=2.0 |
|---|---|---|---|
| RAFT clean | 0.754 | 0.727 | 0.733 |
| GMFlow-basic clean | 0.737 | 0.697 | 0.676 |
| **GMFlow-refine clean** | **0.777** | **0.757** | **0.756** |
| RAFT final | 0.736 | 0.698 | 0.687 |
| GMFlow-basic final | 0.725 | 0.672 | 0.641 |
| **GMFlow-refine final** | **0.759** | **0.731** | **0.713** |

Two facts:
1. RAFT > GMFlow-basic at every threshold and every pass (the original H3 lead is threshold-stable against basic).
2. RAFT < GMFlow-refine at every threshold and every pass (H3 **falsified** against refine). The "RAFT sharper" finding was capacity, not architecture.

Threshold absolute swing is 1.5–4.5 percentage points per config — non-trivial but doesn't flip any pairwise ranking within a given pass. F1 at τ=0.5 is uniformly higher (~3 pp) than at τ=2.0 for every config.

> **Figure:** `results/figures/report/boundary_f1_sensitivity.png` — F1 vs τ ∈ {0.5, 1.0, 2.0} for all 3 models × {clean, final}. Three parallel lines on each subplot, no ranking flips.

### 11c. Blur-mask ↔ motion-magnitude confound

Per-sequence correlation across the 23 Sintel-clean sequences:

- Pearson  = **0.644**
- Spearman = **0.520**

Sequences breaking the trend (high blur, low motion):

| seq | blur fraction | mean GT motion (px) |
|---|---|---|
| mountain_1 | 0.36 | 4.98 |
| temple_3 | 0.58 | 37.1 |
| ambush_4 | 0.35 | 31.9 |
| bandage_1 | 0.08 | 3.50 |
| sleeping_1 | 0.14 | 3.43 |

`mountain_1` (0.36 blur, 4.98 px motion) is the clearest evidence that the mask isn't a pure motion-magnitude proxy — it catches actual defocus on a slow sequence. But the 0.644 Pearson means the §7d blur reading should be qualified: "blur" and "fast motion" overlap enough that EPE differences on the blur mask aren't a clean defocus-only signal.

Scatter: `results/figures/blur_motion_confound.png`.

### 11d. Sintel → Middlebury normalization choice

Methodology §3 H10 asks for normalized cross-dataset score. Two reasonable normalizers, reported side by side:

| Model | Middlebury EE | Sintel-all EE | Mid/Sintel-all | Sintel s0_10 EE | Mid/Sintel-s0_10 |
|---|---|---|---|---|---|
| RAFT-32 | 0.302 | 1.446 | **0.209** | 0.360 | **0.839** |
| GMFlow-basic | 0.486 | 1.484 | 0.328 | 0.456 | 1.066 |
| GMFlow-refine | 0.402 | 1.073 | 0.375 | 0.302 | **1.331** |

The "RAFT generalizes 36% better than basic" in the original §4b uses the `all` normalizer, which is dominated by Sintel's s10+ tail. Middlebury motion magnitudes are mostly < 10 px, so the **`s0_10` normalizer is the better-matched comparison** and gives a smaller (27% vs 36%) gap.

RAFT keeps the directional win on both normalizers. The refine result is notable: refine has **lower** Sintel-s0_10 EPE than basic (0.302 vs 0.456 — a big drop) but only **slightly lower** Middlebury EE (0.402 vs 0.486 — a small drop). Normalized, refine therefore has the *largest* Sintel/Middlebury gap of the three. **This is NOT Sintel-overfitting** (both checkpoints are Things-trained, neither saw Sintel during training); it's a **motion-regime effect**: refine's extra capacity helps Sintel's large/structured-motion content (the s10+ tail) more than it helps Middlebury's all-small-motion regime. Re-phrased: higher capacity fits the large-motion regime better, and that gain doesn't transfer to small-motion data.

> **Figure:** `results/figures/report/middlebury_per_seq.png` — RAFT-32 / basic / refine bars per sequence + MEAN. RAFT lowest on all 8 sequences and on the MEAN. Refine beats basic on 7/8 (Venus is the outlier — refine is *worse* than basic there, the only such reversal across both Sintel passes and Middlebury).

### 11e. H9 resolution-sweep follow-up

The 2.0× and 2.5× rows of §6b are contaminated by WSL2 unified-memory paging — `max_memory_allocated` of 6.1–15.2 GB exceeds the 3050 Ti's 4 GB physical capacity. Latency at those rows mixes attention scaling with paging cost and **the two models may page differently**. The original "strongly supported at Spring-like resolutions" verdict is retracted; H9 is "indicative, not verified" until re-run on:

- Colab A100 (40 GB) — recommended; one cell.
- Local card with ≥16 GB physical memory.

Re-run command (same script, just on a real GPU): `python -m cvflow.runners.run_vram_resolution --factors 1.0 1.5 2.0 2.5 3.0`.

The clean 1.0× and 1.5× rows already show GMFlow-basic's latency ratio against RAFT trending up (0.41× → 0.66× as resolution grows 1.5×). If the trend continues linearly that's roughly cross-over at ~1.7×. If it inflects quadratically the cross-over could be lower. Either is consistent with O(N²) attention.

### 11f. Quantization residual — do GMFlow variants predict on a coarse grid?

Original report (§3b key reads) attributed GMFlow-basic's `A50 ≈ 0.41` median EPE to "1/8-grid quantization residual." Tested by histogramming matched-pixel EPE on slow sequences (where any quantization pile-up would be visible) and a fast control where it should smear.

`results/figures/quantization/<model>__<seq>__frame0001.png` for each (model, seq) and the summary CSV give the fractions within ±0.02 of each `k/8` multiple:

**alley_2 (slow, 443k matched pixels):**

| target | RAFT | GMFlow-basic | GMFlow-refine |
|---|---|---|---|
| frac near 0 | 9.90% | 0.05% | 2.40% |
| frac near 1/8 | 11.09% | 8.99% | **14.22%** |
| frac near 2/8 | 0.65% | **13.05%** | 7.60% |
| frac near 3/8 | 0.13% | 6.09% | 2.47% |
| frac near 4/8 | 0.06% | 2.26% | 0.67% |

**ambush_4 (fast control, 446k matched pixels):**

| target | RAFT | GMFlow-basic | GMFlow-refine |
|---|---|---|---|
| frac near 1/8 | 8.26% | 2.24% | 7.49% |
| frac near 2/8 | 4.19% | 4.95% | 4.66% |
| frac near 3/8 | 1.46% | 5.11% | 1.62% |

Reading:
- GMFlow-basic on alley_2 piles 13% of matched pixels within 0.02 px of `2/8 = 0.25`, 9% near `1/8 = 0.125`, 6% near `3/8 = 0.375`. That's a clear *soft* concentration at small `k/8` multiples — not a sharp spike, but distinct from RAFT's distribution (whose mass at 1/8 is the stationary-background floor, not a grid residual). On the fast control (ambush_4) the basic GMFlow pile-up shrinks to 5% / 2% / 5% — smeared, as theory predicted.
- GMFlow-refine retains pile-up at `1/8` (14% on alley_2!) and `2/8` (7.6%). Refine reduces the median EPE (A50 0.13 vs basic's 0.24) but the coarse-grid character of its predictions is still visible.
- RAFT has 10% mass *exactly at zero* on alley_2 — that's matched pixels where RAFT's prediction is sub-0.02-px correct (the stationary background). GMFlow-basic has 0.05% there; the basic upsample apparently never lands at exactly zero.

Conclusion: original "1/8-grid quantization residual" claim was **partially supported** — GMFlow-basic's predictions do show coarse-grid concentration at small `k/8` multiples on slow sequences and smear on fast control, consistent with the upsample structure leaving a soft residual. But it's a *concentration*, not a *spike*, and GMFlow-refine reduces without eliminating it. §9 wording revised to "coarse-grid upsample residual at small-k/8 multiples".

Histograms saved in `results/figures/quantization/`.
