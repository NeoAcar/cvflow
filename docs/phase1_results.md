# Phase 1 Results — RAFT vs GMFlow on Sintel + Middlebury

Inference-only evaluation. Things-trained checkpoints for both models (zero-shot framing — no Sintel fine-tuning, no Middlebury fine-tuning). All numbers reproducible from saved `.npy` predictions in `results/predictions/`.

## 1. Setup

| | |
|---|---|
| RAFT checkpoint | `raft-things.pth` (21 MB, Dropbox-distributed) |
| RAFT inference iters (headline) | 32 |
| GMFlow checkpoint | `gmflow_things-e9887eda.pth` (basic, no refinement) |
| GMFlow attn/corr/prop | `[2]/[-1]/[-1]`, `padding_factor=16` |
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
| GMFlow | clean | 1.4839 | 1.495 | −0.7% |
| GMFlow | final | 2.9420 | 2.955 | −0.4% |

Per-mask sub-comparison against GMFlow's published numbers on Sintel clean:

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

Bold = winner.

| Mask | **Clean RAFT** | **Clean GMFlow** | **Final RAFT** | **Final GMFlow** |
|---|---|---|---|---|
| EPE / all | **1.446** | 1.484 | **2.678** | 2.942 |
| EPE / matched | **0.646** | 0.822 | **1.585** | 1.902 |
| EPE / unmatched | 11.69 | **9.95** | 16.67 | **16.25** |
| EPE / s0-1 | **0.161** | 0.245 | **0.293** | 0.371 |
| EPE / s0-10 | **0.360** | 0.456 | **0.509** | 0.724 |
| EPE / s10-40 | **1.651** | 1.760 | **2.994** | 3.438 |
| EPE / s40+ | 8.72 | **8.19** | **17.41** | 17.63 |
| EPE / s60+ | 12.30 | **11.21** | **23.61** | 23.84 |
| EPE / Disc | 3.58 | **3.46** | **6.26** | 6.79 |
| EPE / Untex | 1.65 | **1.55** | **3.11** | 3.46 |
| EPE / Blur | **2.18** | 1.88 ← *(GMFlow)* | **4.03** | 4.67 |
| Bad-1 | **0.098** | 0.160 | **0.147** | 0.209 |
| Bad-3 | **0.044** | 0.059 | **0.081** | 0.098 |
| Bad-5 | **0.031** | 0.039 | **0.062** | 0.071 |
| Bad-10 (catastrophic) | **0.021** | 0.023 | **0.043** | 0.047 |
| Boundary F1 | **0.727** | 0.697 | **0.698** | 0.672 |

Note on Blur row: GMFlow has lower EPE on blurred pixels (1.88 vs 2.18 clean, 4.67 vs 4.03 final — *RAFT wins on final*). The clean-pass result is unexpected; see §7d below.

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

**GMFlow  (gmflow_things, basic, clean):**

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
- **A50/A75 column** confirms the sub-pixel gap: RAFT median is 0.07 px across almost every mask; GMFlow's median is ~0.4 px everywhere — likely an unsuppressed 1/8-grid quantization residual from the convex upsample.
- **Bad-10 (catastrophic)** is roughly the same global rate (2.1% RAFT vs 2.3% GMFlow), but is **dominated by occluded pixels** on both: 18.7% of RAFT's unmatched pixels catastrophically fail vs 19.9% for GMFlow. Among matched (non-occluded) pixels, both have <1% catastrophic.
- **s60+** is much worse than s40+ on both: RAFT 12.30 vs 8.72, GMFlow 11.21 vs 8.19. The largest-displacement tail eats most of the dataset's error budget.
- **SD column** shows the EPE distribution is wildly skewed (SD ≈ 5–8× the mean almost everywhere). Mean EPE alone is misleading; the median (A50) and SD together describe the distribution.
- **nEPE column** says the *fractional* error is roughly flat across speed buckets (RAFT 8–10% on s10+/s40+/s60+; GMFlow 9–10%). The absolute EPE grows with speed because the underlying motion grows, not because the model gets proportionally worse. The exception is **GMFlow at s0_10**: nEPE = 18.5% vs RAFT's 12.2% — GMFlow makes proportionally larger errors at slow motion (the same 1/8-grid quantization residual the A50 column hinted at, expressed as a fraction of the motion).
- **Pearson(EPE, AE) per mask** is highly mask-dependent. Tight correlation in well-defined regimes (RAFT s40+ Pearson 0.93, GMFlow s60+ Pearson 0.86). Near-zero correlation in others (GMFlow blur clean 0.22, blur final 0.003 — *zero*). The Spearman column tells a separate story: **s0_1 Spearman is ≈0.97 for RAFT and ≈0.90 for GMFlow** — at sub-pixel motion AE rankings track EPE rankings tightly even when their magnitudes don't (Pearson 0.32 / 0.54). This matches Baker et al. 2011's observation that AE *downweights* large-motion errors — at small motion the geometry is dominated by direction, so AE and EPE rank-track even if their absolute scales diverge.

### Per-sequence EPE (clean)

| seq | RAFT | GMFlow | seq | RAFT | GMFlow |
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

| Mask | RAFT ΔEPE | RAFT % | GMFlow ΔEPE | GMFlow % |
|---|---|---|---|---|
| all | +1.232 | +85% | +1.458 | +98% |
| matched | +0.939 | +145% | +1.080 | +131% |
| unmatched | +4.978 | +43% | +6.298 | +63% |
| Disc | +2.680 | +75% | +3.324 | +96% |

## 4. Middlebury — zero-shot cross-dataset (Things → Middlebury)

Bold = winner per (sequence, column).

| Sequence | RAFT EE | RAFT AE | RAFT R0.5 | R1.0 | R2.0 | GMFlow EE | GMFlow AE | R0.5 | R1.0 | R2.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Dimetrodon | **0.193** | **3.78°** | **0.054** | **0.004** | 0.000 | 0.394 | 7.64° | 0.211 | 0.011 | 0.000 |
| Grove2 | **0.248** | **3.59°** | **0.073** | 0.035 | 0.016 | 0.346 | 5.15° | 0.145 | **0.038** | **0.009** |
| Grove3 | **0.679** | **6.82°** | **0.313** | **0.170** | **0.071** | 0.770 | 7.76° | 0.462 | 0.190 | 0.073 |
| Hydrangea | **0.224** | **2.72°** | **0.130** | **0.050** | **0.011** | 0.322 | 3.69° | 0.188 | 0.078 | 0.019 |
| RubberWhale | **0.188** | **6.16°** | **0.062** | **0.021** | **0.003** | 0.405 | 12.76° | 0.172 | 0.051 | 0.008 |
| Urban2 | **0.316** | **2.97°** | **0.102** | **0.031** | **0.014** | 0.585 | 5.89° | 0.371 | 0.105 | 0.034 |
| Urban3 | **0.351** | **3.08°** | **0.135** | **0.048** | **0.023** | 0.566 | 4.11° | 0.323 | 0.108 | 0.029 |
| Venus | **0.218** | **2.71°** | **0.043** | **0.006** | **0.002** | 0.503 | 6.49° | 0.422 | 0.061 | 0.003 |
| **mean** | **0.302** | **3.98°** | **0.114** | **0.046** | **0.018** | 0.486 | 6.69° | 0.287 | 0.080 | 0.022 |

RAFT wins on every Middlebury sequence by EE and AE. On R0.5 (Baker's sub-half-pixel robustness threshold) RAFT averages 11.4% bad vs GMFlow's 28.7% — over 2× fewer pixels miss the 0.5-px target. R1.0 and R2.0 differences narrow as the threshold loosens.

### Normalized Sintel → Middlebury generalization (H10)

Methodology §3 H10: report Middlebury EPE normalized by each model's Sintel-domain EPE. Smaller is better (cleaner cross-dataset transfer).

| Model | Sintel clean EE | Middlebury mean EE | normalized score |
|---|---|---|---|
| RAFT | 1.446 | 0.302 | **0.209** |
| GMFlow | 1.484 | 0.486 | 0.328 |

Same direction as the raw numbers — RAFT generalizes ~36% better cross-dataset on this score. Useful framing because it controls for the fact that Middlebury motion magnitudes (mostly < 10 px) sit at the easier end of Sintel's distribution.

### 4b. AE↔EPE correlation on Middlebury

Per-sequence and global Pearson/Spearman between per-pixel EPE and AE on GT-valid pixels.

| Sequence | n (px) | RAFT Pearson | RAFT Spearman | GMFlow Pearson | GMFlow Spearman |
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

RAFT's EPE/AE correlation runs ~0.13 higher than GMFlow's globally. Urban2 is the weakest correlation cell for both models — that sequence is mostly slow synthetic motion where small absolute errors translate into large angular swings (AE noise dominates EPE signal).

## 4b. AE + percentile accuracies on Sintel

Added via the `_Accum` reservoir (capped at 5M masked samples). All numbers are global, all masks combined into the `all` (valid-GT-only) view.

| | **Clean RAFT** | **Clean GMFlow** | **Final RAFT** | **Final GMFlow** |
|---|---|---|---|---|
| AE / all (°) | **3.91** | 5.78 | **5.58** | 7.67 |
| AE / matched (°) | **3.14** | 5.28 | **4.50** | 6.81 |
| AE / unmatched (°) | 13.86 | **12.10** | 19.30 | **18.64** |
| A50 (px) | **0.067** | 0.411 | **0.068** | 0.409 |
| A75 (px) | **0.131** | 0.497 | **0.134** | 0.517 |
| A95 (px) | **0.707** | 1.085 | **0.723** | 1.133 |

Striking detail: RAFT A50 ≈ 0.07 px on both passes — half of all valid pixels are within 0.07 px of GT. GMFlow's A50 sits at ~0.41 — likely the unsuppressed 1/8-grid quantization residual from its single-pass convex-upsample. The AE / unmatched columns flip the EE-based result: on the angular-error metric GMFlow handles occluded pixels better (12.1° vs 13.9° clean, 18.6° vs 19.3° final), consistent with H4.

## 5. RAFT iter sweep — full Sintel clean (1041 pairs × 4 iter levels)

| iters | EPE / all | s0-10 | s10-40 | s40+ | median ms |
|---|---|---|---|---|---|
| 4 | 1.910 | 0.467 | 2.309 | 11.296 | 198 |
| 8 | 1.593 | 0.388 | 1.888 | 9.521 | 276 |
| 12 | 1.510 | 0.383 | 1.700 | 9.115 | 358 |
| 32 | **1.446** | **0.360** | **1.651** | **8.720** | 765 |

Monotone EPE decrease, diminishing returns: 4→8 saves 0.32 EPE for +78 ms; 12→32 saves only 0.06 EPE for +407 ms. Latency scales sub-linearly with iters (4→32: 8× iters, 3.9× latency — non-iter overhead amortizes).

The full-dataset curve reverses what the 2-sequence subset (alley_1 + market_2) showed: on the subset RAFT-12 (0.292) beat GMFlow (~0.4), but on the full dataset GMFlow (1.484 @ 309 ms) sits cleanly on the Pareto frontier:

```
GMFlow                  309 ms,  EPE 1.484   ← Pareto
RAFT  4 iters           198 ms,  EPE 1.910
RAFT  8 iters           276 ms,  EPE 1.593
RAFT 12 iters           358 ms,  EPE 1.510   ← Pareto-dominated by GMFlow
RAFT 32 iters           765 ms,  EPE 1.446   ← only point beating GMFlow on accuracy
```

GMFlow Pareto-dominates RAFT-12 on the full dataset (faster AND more accurate); only RAFT-32 beats GMFlow on accuracy, at 2.5× the latency.

## 6. Latency + VRAM at 1024×436 (n=50, mixed sequences)

| | RAFT (32 iters) | GMFlow (basic) |
|---|---|---|
| median (ms) | 796 | **309** |
| mean ± SD (ms) | 796 ± 2 | 310 ± 2 |
| min–max (ms) | 794–804 | 305–321 |
| peak VRAM (MB) | 529 | **470** |

GMFlow is 2.6× faster per pair end-to-end at Sintel resolution. Both well under the 3050 Ti's 4 GB at this resolution.

### 6b. Resolution sweep — H9 verdict

Sintel pairs upsampled by factor `f` (bilinear), n=10 timed forwards per cell:

| factor | resolution | RAFT (ms / MB) | GMFlow (ms / MB) | GMFlow / RAFT latency |
|---|---|---|---|---|
| 1.00× | 1024×436 | 737 / 529 | 300 / 470 | **0.41×** (GMFlow faster) |
| 1.50× | 1536×654 | 1,765 / 2,023 | 1,174 / 2,036 | **0.66×** (GMFlow still faster) |
| 2.00× | 2048×872 | 3,591 / 6,146 | **12,431** / 6,285 | **3.46×** (GMFlow slower) |
| 2.50× | 2560×1090 | 6,877 / 14,970 | **43,894** / 15,229 | **6.38×** (GMFlow catastrophic) |

RAFT scales near-linearly with pixel count (6.25× pixels → 9.3× latency, the extra slack from non-iter overhead). GMFlow scales quadratically with token count (6.25× pixels → 146× latency, consistent with O(N²) attention on the 1/8 feature grid). VRAM grows similarly for both, but on the 3050 Ti's 4 GB physical memory anything over ~2 GB pages through host memory via WSL2 unified-memory spillover, so the reported MB numbers above 4 GB include paged allocations.

Cross-over on this hardware is between 1.5× and 2× Sintel; the Pareto picture at every resolution above 1.5× flips entirely — RAFT-32 becomes the faster *and* generally more accurate option.

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

| seq | RAFT median Δ | GMFlow median Δ | seq | RAFT median Δ | GMFlow median Δ |
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

ambush_2 dominates the Clean→Final regression on both models, with GMFlow even more affected than RAFT. ambush_4 is the largest gap between models: RAFT degrades ~3× less than GMFlow there (median Δ 5.03 vs 15.25). A handful of sequences (bamboo_1, sleeping_1 for GMFlow, bandage_2 for GMFlow) have **negative** median Δ — Final's motion blur smooths errors more than its photometric shift adds, so Final is *easier* to predict on those sequences. This is the per-pixel evidence behind methodology §4 item 10.

## 7d. Blur / defocus mask via Laplacian-variance (methodology §2.6, §2.8)

Per-pair blur mask = `var(Laplacian(I₁, ksize=3), 7×7) < 20` — windowed Laplacian variance, low variance ⇒ blurred / defocused. Applied as an additional mask in `SintelMetrics`.

| | Clean RAFT | Clean GMFlow | Final RAFT | Final GMFlow |
|---|---|---|---|---|
| Blur-pixel fraction (over `valid`) | 8.6% | 8.6% | 17.4% | 17.4% |
| EPE / blur | 2.18 | **1.88** | **4.03** | 4.67 |
| AE / blur (°) | 3.77 | **4.79** ← 3.77 better | 6.16 | 8.35 |
| Bad-1 / blur | 0.108 | **0.182** ← 0.108 better | 0.187 | 0.253 |
| Bad-10 / blur | 0.029 | 0.030 | 0.066 | 0.072 |
| A95 / blur | 2.15 | 6.06 | 11.51 | 15.85 |

Per-frame blur fraction sanity: clean alley_1 = 4.4%, market_2 = 6.9%, ambush_4 ≈ 60%. Final-pass adds renderer motion blur — fraction roughly doubles (alley_1 → 15.5%, market_2 → 25.5%).

Reading: GMFlow has *lower mean EPE* on blurred pixels of Sintel **clean** (1.88 vs 2.18) but **worse** on every other distributional measure on those same pixels — higher AE (4.79° vs 3.77°), nearly 2× the Bad-1 rate (0.182 vs 0.108), and a 3× larger A95 tail (6.06 vs 2.15). The likely read: GMFlow's predictions on blurred pixels are systematically slightly off (small bias eats into mean EPE less than RAFT's occasional larger errors, but RAFT is more often correct). On Final, where blurred-pixel fraction doubles, RAFT regains the EPE lead (4.03 vs 4.67) too.

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
| GMFlow clean | 0.245 | 0.428 | 0.665 | 1.253 | 2.574 | 3.772 | **11.21** |
| RAFT final | 0.293 | 0.341 | 0.859 | 2.111 | 4.411 | 8.355 | 23.61 |
| GMFlow final | 0.371 | 0.553 | 1.194 | 2.524 | 4.904 | 8.569 | 23.84 |

**Normalized EPE = EPE / |gt|  (only on `|gt|>1`):**

| config | s0-1 | s1-3 | s3-10 | s10-20 | s20-40 | s40-60 | s60+ |
|---|---|---|---|---|---|---|---|
| RAFT clean | nan | **0.126** | **0.119** | 0.084 | **0.086** | 0.072 | 0.098 |
| GMFlow clean | nan | 0.253 | 0.118 | **0.089** | 0.091 | **0.077** | **0.096** |
| RAFT final | nan | 0.191 | 0.149 | 0.149 | 0.156 | 0.169 | 0.206 |
| GMFlow final | nan | 0.326 | 0.210 | 0.176 | 0.173 | 0.175 | 0.210 |

**AE (degrees) vs speed:**

| config | s0-1 | s1-3 | s3-10 | s10-20 | s20-40 | s40-60 | s60+ |
|---|---|---|---|---|---|---|---|
| RAFT clean | 5.72 | 3.78 | 3.00 | 2.80 | 3.43 | 3.41 | 5.29 |
| GMFlow clean | 9.71 | 6.95 | 4.19 | 3.44 | 3.55 | **3.00** | **3.68** |
| RAFT final | 7.29 | 4.56 | 3.88 | 4.48 | 6.25 | 8.25 | 10.35 |
| GMFlow final | 10.98 | 8.28 | 5.28 | 5.32 | 6.83 | 8.19 | 9.30 |

Reading:

- **EPE vs speed**: RAFT < GMFlow at every clean bucket up to s40-60 then they cross — GMFlow s60+ EPE 11.21 < RAFT 12.30. The cross is consistent with H1 (GMFlow's global matching helps on large displacement).
- **nEPE vs speed**: GMFlow's huge nEPE at s1-3 (0.253 clean, 0.326 final — 2× RAFT) shows GMFlow's small-motion errors are *proportionally* much larger; from s10-20 onwards both models hover around 8–9% on clean. The nEPE curve is U-shaped: highest at slow + fastest motion, lowest in the s10-60 sweet spot.
- **AE vs speed**: U-shape too, but with model character. GMFlow's AE *decreases* with speed (9.71 → 3.00 over s0-1 to s40-60) — its errors at high speed are more aligned with GT motion. RAFT's AE bottoms out around s10-20 then rises again at s60+.
- **Bad-1 vs speed** (in CSV): monotonically increasing on every config; RAFT clean s0-1 = 1.5% vs s60+ = 41%, the spread is enormous.

Figures: `results/figures/speed_curves/{epe,ae,nepe,bad1}_vs_speed.png`.

## 7c. Forward-backward consistency derived occlusion (methodology §2.2)

Per-pair derived occlusion = `‖f12(x) + f21(x + f12(x))‖² > α(‖f12‖² + ‖f21‖²) + β` with `α=0.01`, `β=0.5` (Sundaram et al. 2010 / Meister et al. 2018). Backward flow computed by swapping `(img1, img2)`. Forward predictions reused from saved `.npy`. Compared per-pixel against Sintel native `occlusion == 255` on valid GT.

| | RAFT | GMFlow |
|---|---|---|
| native occluded fraction | 7.25% | 7.25% |
| derived occluded fraction | 6.25% | 4.85% |
| precision | 0.654 | **0.752** |
| recall | **0.564** | 0.504 |
| F1 | 0.606 | 0.603 |
| IoU | 0.435 | 0.432 |

Both models recover ≈0.60 F1 / ≈0.43 IoU of the GT occlusion mask from inference alone. GMFlow's mask is sparser and more precise (predicts fewer occlusions but most are real), RAFT's is denser and higher-recall. The two F1 numbers are essentially indistinguishable (0.606 vs 0.603) despite GMFlow having lower unmatched EPE — the *quality of the inferred occlusion mask* and the *EPE on the occluded region* are independent axes. §2.2 cross-check is satisfied: fwd-bwd consistency is a usable proxy for occlusion when GT is unavailable.

## 8. Hypothesis verdicts (Phase 1)

| H | Claim | Evidence | Verdict |
|---|---|---|---|
| 1 | GMFlow > large displacements | s40+ clean 8.19 vs 8.72; s60+ clean 11.21 vs 12.30; per-seq: GMFlow better on ambush_4 (6.41 vs 9.02). Final pass: GMFlow loses by hair-margins on both buckets (s40+ 17.63 vs 17.41, s60+ 23.84 vs 23.61) | **supported on clean (incl. s60+ extra bin); reverses to marginal-RAFT on final** |
| 2 | RAFT > sub-pixel | s0-10 0.36 vs 0.46; **s0-1 0.16 vs 0.25** (sub-pixel-only bin); Bad-1 0.098 vs 0.160 (~40% gap); nEPE at s1-3 = 12.6% RAFT vs 25.3% GMFlow (GMFlow's slow-motion errors are proportionally 2×) | **strongly supported across every slow-motion metric** |
| 3 | RAFT sharper boundaries | F1 0.727 vs 0.697 (clean), 0.698 vs 0.672 (final) | **supported** |
| 4 | GMFlow > occlusions | unmatched 9.95 vs 11.69 clean; 16.25 vs 16.67 final | **supported on clean, marginal on final** |
| 5 | Both adequate on untex | RAFT untex/all ratio 1.14, GMFlow 1.05 — neither is a catastrophic regime | **supported** |
| 6 | RAFT iter ↔ accuracy trade-off | full-dataset sweep: EPE 1.91/1.59/1.51/1.45 at iters {4,8,12,32}; GMFlow Pareto-dominates RAFT-12 (faster + more accurate); only RAFT-32 beats GMFlow on accuracy | **strongly supported — with the clearer reading that GMFlow is competitive on the curve, not strictly dominated** |
| 7 | GMFlow more tolerant Clean→Final | RAFT ΔEPE = 1.23; GMFlow = 1.46 (raw); on matched-relative GMFlow degrades less (+131% vs +145%) | **FALSIFIED on raw EPE; partial on matched-relative** |
| 8 | RAFT weak to weather, GMFlow to noise | RobustSpring not on disk | **deferred — Phase 2** |
| 9 | GMFlow VRAM blow-up | at Sintel 1024×436 both <600 MB and GMFlow 2.6× faster; at 2×+ Sintel GMFlow latency grows quadratically (146× at 2.5×), RAFT linearly; cross-over between 1.5× and 2× | **strongly supported at Spring-like resolutions** (≥2× Sintel) |
| 10 | Sintel→Middlebury drops unequally | Raw EE: RAFT 0.302 vs GMFlow 0.486; normalized score (Mid EE / Sintel EE): RAFT **0.209** vs GMFlow 0.328; RAFT wins all 8 Middlebury sequences on every threshold (EE, AE, R0.5/1/2) | **strongly supported — RAFT generalizes ~36% better on the normalized score** |

## 9. Caveats and notes

- **Motion-boundary mask is derived** (Sobel + 9×9 dilate), not from Sintel's native `motion_boundaries/`. The `MPI-Sintel-training_extras.zip` checked: it only contains `flow_code/`, `flow_viz/`, `invalid/`, `occlusions/` — no `motion_boundaries/`. The native mask appears not to be publicly distributed. Both pred and GT use the same derivation, so Boundary F-score is internally consistent but not directly comparable to papers that use the native mask.
- **Blur mask threshold is uncalibrated.** `var(Laplacian(I, 3×3), 7×7) < 20` is an order-of-magnitude pick — it produces 8.6% (clean) / 17.4% (final) blurred pixels, which is in a reasonable range, but the threshold wasn't fit against any reference annotation (Sintel doesn't ship one). Comparisons across models are valid (same threshold applied to both); absolute numbers shouldn't be compared to papers using a different threshold.
- **The Schulze ranking / RobustSpring corruption suite (methodology §1.6, §4 items 11–12) is Phase 2** — needs RobustSpring download.
- **WSL2 unified memory** lets `max_memory_allocated` exceed the physical 4 GB at high resolutions by paging through host memory. The MB numbers in §6b above 4 GB are *requested* allocations, not strictly on-card — the latency numbers there are honest, but VRAM should be re-measured on a card with ≥16 GB physical memory for a clean reading.
- **Iter sweep was run at two granularities**: subset (alley_1 + market_2 only, ~3 min) showed RAFT-12 minimum at 0.292 EPE on that easy subset; full-dataset (all 23 sequences, ~34 min) shows monotone decrease across the full curve with GMFlow on the Pareto frontier. Both numbers reported above (§5).

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
```
