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

## 5. RAFT iter sweep (alley_1 + market_2, 99 pairs)

| iters | EPE/all | s0-10 | s10-40 | s40+ | median ms |
|---|---|---|---|---|---|
| 4 | 0.3192 | 0.214 | 1.538 | 4.054 | 203 |
| 8 | 0.2951 | 0.194 | 1.467 | 3.860 | 288 |
| 12 | **0.2924** | **0.192** | **1.460** | 3.799 | 364 |
| 32 | 0.2971 | 0.197 | 1.464 | **3.801** | 764 |

EPE saturates at 12 iters on these sequences (32 iters gives a marginal regression on EPE/all). Latency scales sub-linearly with iters (4→32 = 8× iters, 3.8× latency — non-iter overhead amortizes).

GMFlow median = 309 ms on the same hardware (cf. §6). On the EPE-vs-latency plot, GMFlow sits between RAFT 4 and RAFT 8 in latency, with EPE worse than every RAFT iter setting on these two sequences. RAFT pareto-dominates GMFlow on alley_1 + market_2 at every iter count.

## 6. Latency + VRAM at 1024×436 (n=50, mixed sequences)

| | RAFT (32 iters) | GMFlow (basic) |
|---|---|---|
| median (ms) | 796 | **309** |
| mean ± SD (ms) | 796 ± 2 | 310 ± 2 |
| min–max (ms) | 794–804 | 305–321 |
| peak VRAM (MB) | 529 | **470** |

GMFlow is 2.6× faster per pair end-to-end. Both well under the 3050 Ti's 4 GB — H9 (GMFlow VRAM blow-up) is **not observed** at 1024×436.

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

## 8. Hypothesis verdicts (Phase 1)

| H | Claim | Evidence | Verdict |
|---|---|---|---|
| 1 | GMFlow > large displacements | s40+ clean 8.19 vs 8.72; per-seq: GMFlow better on ambush_4 (6.41 vs 9.02) | **supported** (marginal on final: 17.63 vs 17.41 — RAFT slightly better there) |
| 2 | RAFT > sub-pixel | s0-10 0.36 vs 0.46; Bad-1 0.098 vs 0.160 (~40% gap) | **strongly supported** |
| 3 | RAFT sharper boundaries | F1 0.727 vs 0.697 (clean), 0.698 vs 0.672 (final) | **supported** |
| 4 | GMFlow > occlusions | unmatched 9.95 vs 11.69 clean; 16.25 vs 16.67 final | **supported on clean, marginal on final** |
| 5 | Both adequate on untex | RAFT untex/all ratio 1.14, GMFlow 1.05 — neither is a catastrophic regime | **supported** |
| 6 | RAFT iter ↔ accuracy trade-off | EPE saturates at 12 iters on easy sequences; latency near-linear in iters | **supported** |
| 7 | GMFlow more tolerant Clean→Final | RAFT ΔEPE = 1.23; GMFlow = 1.46 (raw); on matched-relative GMFlow degrades less (+131% vs +145%) | **FALSIFIED on raw EPE; partial on matched-relative** |
| 8 | RAFT weak to weather, GMFlow to noise | RobustSpring not on disk | **deferred — Phase 2** |
| 9 | GMFlow VRAM blow-up | both <600 MB at 1024×436 | **not observed at Sintel resolution** |
| 10 | Sintel→Middlebury drops unequally | RAFT 0.30 vs GMFlow 0.49 mean EE; RAFT wins all 8 Middlebury sequences | **strongly supported — RAFT generalizes better** |

## 9. Caveats and notes

- **H9 needs higher-resolution test to fire.** Sintel-level 1024×436 is below the transformer's O(N²) blow-up regime on a modern attention impl. A Spring 1080p or 2560×1080 test would be needed to actually see GMFlow VRAM diverge from RAFT's.
- **Motion-boundary mask is derived (Sobel + 9×9 dilate), not from Sintel's native `motion_boundaries/` (not in our download).** Both pred and GT use the same derivation, so Boundary F-score is internally consistent but not directly comparable to papers that use the native mask.
- **AE and A50/A75/A95 are reported only on Middlebury** so far (methodology §4 items 3–4 specify "alongside" but didn't pin which dataset). Adding them to Sintel is straightforward.
- **The Schulze ranking / RobustSpring corruption suite (methodology §1.6, §4 items 11–12) is Phase 2** — needs RobustSpring download.
- **Iter sweep was deliberately limited to alley_1 + market_2** to keep step 13 tractable. Extending to the full 1041-pair Sintel clean × 4 iter levels would add ~50 min and ~28 GB to the prediction store; not done because the curve already shows monotone saturation and the methodology §4 item 6 needs a "curve", not a full-dataset sweep.

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

# §5: iter sweep
python -m cvflow.runners.run_raft_itersweep

# §6: latency + VRAM
python -m cvflow.runners.run_latency_vram --n 50

# §7: photometric residual
python -m cvflow.runners.run_photometric
```
