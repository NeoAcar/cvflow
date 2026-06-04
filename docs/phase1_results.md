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
| EPE / s0-10 | **0.360** | 0.456 | **0.509** | 0.724 |
| EPE / s10-40 | **1.651** | 1.760 | **2.994** | 3.438 |
| EPE / s40+ | 8.72 | **8.19** | **17.41** | 17.63 |
| EPE / Disc | 3.58 | **3.46** | **6.26** | 6.79 |
| EPE / Untex | 1.65 | **1.55** | **3.11** | 3.46 |
| Bad-1 | **0.098** | 0.160 | **0.147** | 0.209 |
| Bad-3 | **0.044** | 0.059 | **0.081** | 0.098 |
| Bad-5 | **0.031** | 0.039 | **0.062** | 0.071 |
| Boundary F1 | **0.727** | 0.697 | **0.698** | 0.672 |

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

| Sequence | RAFT EE | RAFT AE | GMFlow EE | GMFlow AE |
|---|---|---|---|---|
| Dimetrodon | **0.193** | 3.78° | 0.394 | 7.64° |
| Grove2 | **0.248** | 3.59° | 0.346 | 5.15° |
| Grove3 | **0.679** | 6.82° | 0.770 | 7.76° |
| Hydrangea | **0.224** | 2.72° | 0.322 | 3.69° |
| RubberWhale | **0.188** | 6.16° | 0.405 | 12.76° |
| Urban2 | **0.316** | 2.97° | 0.585 | 5.89° |
| Urban3 | **0.351** | 3.08° | 0.566 | 4.11° |
| Venus | **0.218** | 2.71° | 0.503 | 6.49° |
| **mean** | **0.302** | **3.98°** | 0.486 | 6.69° |

RAFT wins on every Middlebury sequence by both EE and AE.

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
| 1 | GMFlow > large displacements | s40+ clean 8.19 vs 8.72; per-seq: GMFlow better on ambush_4 (6.41 vs 9.02) | **supported** (marginal on final: 17.63 vs 17.41 — RAFT slightly better there) |
| 2 | RAFT > sub-pixel | s0-10 0.36 vs 0.46; Bad-1 0.098 vs 0.160 (~40% gap) | **strongly supported** |
| 3 | RAFT sharper boundaries | F1 0.727 vs 0.697 (clean), 0.698 vs 0.672 (final) | **supported** |
| 4 | GMFlow > occlusions | unmatched 9.95 vs 11.69 clean; 16.25 vs 16.67 final | **supported on clean, marginal on final** |
| 5 | Both adequate on untex | RAFT untex/all ratio 1.14, GMFlow 1.05 — neither is a catastrophic regime | **supported** |
| 6 | RAFT iter ↔ accuracy trade-off | full-dataset sweep: EPE 1.91/1.59/1.51/1.45 at iters {4,8,12,32}; GMFlow Pareto-dominates RAFT-12 (faster + more accurate); only RAFT-32 beats GMFlow on accuracy | **strongly supported — with the clearer reading that GMFlow is competitive on the curve, not strictly dominated** |
| 7 | GMFlow more tolerant Clean→Final | RAFT ΔEPE = 1.23; GMFlow = 1.46 (raw); on matched-relative GMFlow degrades less (+131% vs +145%) | **FALSIFIED on raw EPE; partial on matched-relative** |
| 8 | RAFT weak to weather, GMFlow to noise | RobustSpring not on disk | **deferred — Phase 2** |
| 9 | GMFlow VRAM blow-up | at Sintel 1024×436 both <600 MB and GMFlow 2.6× faster; at 2×+ Sintel GMFlow latency grows quadratically (146× at 2.5×), RAFT linearly; cross-over between 1.5× and 2× | **strongly supported at Spring-like resolutions** (≥2× Sintel) |
| 10 | Sintel→Middlebury drops unequally | RAFT 0.30 vs GMFlow 0.49 mean EE; RAFT wins all 8 Middlebury sequences | **strongly supported — RAFT generalizes better** |

## 9. Caveats and notes

- **Motion-boundary mask is derived** (Sobel + 9×9 dilate), not from Sintel's native `motion_boundaries/`. The `MPI-Sintel-training_extras.zip` checked: it only contains `flow_code/`, `flow_viz/`, `invalid/`, `occlusions/` — no `motion_boundaries/`. The native mask appears not to be publicly distributed. Both pred and GT use the same derivation, so Boundary F-score is internally consistent but not directly comparable to papers that use the native mask.
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

# Disc/Untex/F-score offline from saved predictions:
python -m cvflow.runners.eval_from_saved --pred-root results/predictions/raft-raft-things-iter32   --pass clean
python -m cvflow.runners.eval_from_saved --pred-root results/predictions/gmflow-gmflow_things-e9887eda --pass clean

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
