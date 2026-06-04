# Progress

## Done

### Setup
- [x] uv venv (Python 3.11) + PyTorch 2.6.0+cu124
- [x] Confirmed local CUDA (RTX 3050 Ti Laptop, 4 GB) — fallback path is Colab A100
- [x] RAFT checkpoints downloaded to `RAFT/RAFT/models/` (5 .pth, 87 MB total)
- [x] GMFlow checkpoints downloaded to `gmflow/gmflow/pretrained/` (8 .pth, 142 MB total)
- [x] Feasibility check (see `docs/initial_plan.md`)
- [x] Repo layout scaffolded (`src/cvflow/{datasets,models,masks,metrics,runners}`)

### Pipeline (Phase 1)
- [x] `cvflow.flow_io` — `.flo` reader (PIEH magic), PNG mask reader, RGB image reader
- [x] `cvflow.datasets.sintel.Sintel` — 1041-pair iterator with `(img1, img2, gt, occlusion, invalid)`
- [x] `cvflow.datasets.middlebury.Middlebury` — 8-pair iterator (GT-only sequences)
- [x] `cvflow.models.raft_wrapper.RaftWrapper` — RAFT inference, 32 iters headline
- [x] `cvflow.models.gmflow_wrapper.GMFlowWrapper` — GMFlow inference, basic config
- [x] `cvflow.models._padder.InputPadder` — shared padder, avoids `utils.utils` collision
- [x] `cvflow.masks.textureless.untext_mask` — Sobel gradient + 3×3 dilate
- [x] `cvflow.masks.motion_boundary.disc_mask` — Sobel on GT flow + 9×9 dilate
- [x] `cvflow.masks.photometric.photometric_residual` — warp + abs diff
- [x] `cvflow.masks.blur.blur_mask` — windowed Laplacian variance (methodology §2.6, §2.8)
- [x] `cvflow.metrics.sintel.SintelMetrics` — EPE + SD + AE + nEPE (`EPE/|gt|` on `|gt|>1`) + Bad-1/3/5/**10** + A50/A75/A95 + Pearson + Spearman over `{all, matched, unmatched, s0_1, s0_10, s10_40, s40+, s60+, disc, untex, blur}` (every metric on every mask)
- [x] `cvflow.metrics.boundary_fscore.boundary_fscore` — F1 with `tol_px=2`
- [x] `cvflow.metrics.middlebury` — EE, AE, R0.5/R1.0/R2.0, A50/A75/A95 with `|flow|<1e9` mask

### Runners (Phase 1)
- [x] `runners/run_sintel_eval.py` — full per-mask Sintel eval; saves `.npy` predictions
- [x] `runners/eval_from_saved.py` — offline (CPU) Disc/Untex/F-score/AE/A50/A75/A95 from saved `.npy`
- [x] `runners/run_middlebury.py` — Middlebury 8-pair sweep, both models
- [x] `runners/run_raft_itersweep.py` — RAFT iter sweep, configurable `--seqs` and `--iters`
- [x] `runners/run_latency_vram.py` — n=50 timing + peak VRAM at 1024×436
- [x] `runners/run_photometric.py` — photometric residual clean vs final per-sequence
- [x] `runners/delta_epe_maps.py` — per-sequence ΔEPE PNG maps + summary CSV
- [x] `runners/run_vram_resolution.py` — latency + VRAM at upsample factors {1, 1.5, 2, 2.5}
- [x] `runners/run_fwdbwd_occlusion.py` — fwd-bwd consistency mask vs Sintel native (P/R/F1/IoU)
- [x] `runners/speed_curves.py` — fine-grained speed-bucket binning + 4 line plots (EPE / AE / nEPE / Bad-1)
- [x] `runners/middlebury_correlation.py` — per-seq + global Pearson/Spearman AE↔EPE on Middlebury
- [x] `cvflow.analysis.bootstrap` — paired sequence-level bootstrap (10k resamples, returns Δ + 95% CI + p)
- [x] `runners/bootstrap_compare.py` — CI'd Δ tables for any pair of per-seq JSONs
- [x] `runners/boundary_threshold_sweep.py` — F1 sensitivity at τ ∈ {0.5, 1.0, 2.0} for any (tag, pass) set
- [x] `runners/quantization_check.py` — matched-pixel EPE histograms + k/8 pile-up CSV
- [x] `runners/blur_motion_confound.py` — Pearson/Spearman between per-sequence blur fraction and mean GT motion
- [x] GMFlow refine-preset wired via `--gmflow-refine` on `run_sintel_eval` and `run_middlebury`
- [x] `runners/step7_sanity.py` — original §4 item 13 gate runner (kept for history)

### Runs (Phase 1 + follow-ups)
- [x] Sintel **clean** — RAFT (14 min) + GMFlow (6 min), full per-mask + saved `.npy`
- [x] Sintel **final** — RAFT (15 min) + GMFlow (8 min), full per-mask + saved `.npy`
- [x] Disc/Untex/F-score/AE/A50-95 offline pass on all 4 (RAFT/GMFlow × clean/final)
- [x] Middlebury both models (8 pairs each, zero-shot Things-trained)
- [x] RAFT iter sweep `{4, 8, 12, 32}` on alley_1 + market_2 (subset)
- [x] **Full-dataset RAFT iter sweep** all 23 sequences × {4, 8, 12, 32}
- [x] Latency + peak VRAM at 1024×436 (n=50, mixed sequences)
- [x] **Resolution sweep latency+VRAM** at factors {1.0, 1.5, 2.0, 2.5} of Sintel
- [x] Photometric residual per-sequence on clean + final
- [x] **ΔEPE per-pixel maps** for both models (46 PNGs + 2 summary CSVs)
- [x] **Fwd-bwd derived occlusion** vs Sintel native, both models
- [x] **GMFlow-refine** full Sintel clean + final + Middlebury (the with-refinement upstream preset; new model tag `gmflow-gmflow_with_refine_things-36579974`). Sanity passes within 0.5–1.0% of upstream `evaluate.sh` targets.
- [x] **Paired sequence bootstrap** on all three model pairings × {clean, final} × {EPE, AE, Bad-1, Bad-10} × 11 masks; outputs in `results/bootstrap/`
- [x] **Boundary F1 sensitivity sweep** at τ ∈ {0.5, 1.0, 2.0} for all 3 models × 2 passes (`results/figures/boundary_threshold/sensitivity.csv`)
- [x] **Quantization-residual histograms** for all 3 models on slow + fast control sequences (`results/figures/quantization/`)
- [x] **Blur ↔ motion-magnitude confound** Pearson 0.644, Spearman 0.520 (`results/figures/blur_motion_confound.png`)

### Sanity gates passed
- [x] Step 7 gate: RAFT 1.4459 / 2.6779 (target 1.43 / 2.71); GMFlow 1.4839 / 2.9420 (target 1.495 / 2.955) — all ±10%
- [x] GMFlow per-bucket on clean within 0.0–1.8% of `gmflow/scripts/evaluate.sh` published numbers

### Docs
- [x] `docs/initial_plan.md` — feasibility check + repo design + risks
- [x] `docs/phase1_results.md` — full Phase 1 numbers + hypothesis verdicts
- [x] `README.md`
- [x] `CLAUDE.md` updated with project-specific guardrails
- [x] `progress.md` (this file)
- [x] `technical.md`

## Pending

### Out of scope
- ~~Phase 2 (RobustSpring) — methodology §1.6, §4 items 11–12, hypothesis 8~~ **Descoped.** No corruption-robustness numbers will be produced; H8 is marked "not pursued" in §8 of phase1_results.md.

### Phase 1 follow-up findings (all closed)
- AE and A50/A75/A95 added to `SintelMetrics`. RAFT A50=0.067 vs GMFlow A50=0.411 px; RAFT AE/all=3.91° vs GMFlow 5.78° on clean.
- Sintel native `motion_boundaries/` confirmed **not present** in `MPI-Sintel-training_extras.zip` and probably not publicly distributed. Derived Sobel+9×9 stays as the production path.
- Full-dataset RAFT iter sweep (all 23 seqs × {4, 8, 12, 32}): EPE 1.91 / 1.59 / 1.51 / 1.45 at 198 / 276 / 358 / 765 ms. **GMFlow (1.484 EPE @ 309 ms) Pareto-dominates RAFT-12**; only RAFT-32 beats GMFlow on accuracy.
- Resolution sweep — H9 fires at ≥2× Sintel. At 2560×1090: GMFlow 43.9 s/pair vs RAFT-32 6.9 s/pair (quadratic vs linear). VRAM grows similarly for both (~15 GB at 2.5×, paged via WSL2 unified memory).
- ΔEPE maps: 46 PNGs in `results/figures/delta_epe/<tag>/`. ambush_2 worst (RAFT +16 px, GMFlow +21 px median ΔEPE); bamboo_1 and sleeping_1 improve on Final due to motion blur.
- Fwd-bwd derived occlusion vs Sintel native: F1 ~0.60 / IoU ~0.43 for both models. GMFlow higher precision (0.752), RAFT higher recall (0.564). Methodology §2.2 cross-check satisfied.

### Writing
- [x] `docs/phase1_results.md` updated with all follow-up numbers (full-dataset iter sweep, resolution sweep, ΔEPE maps, fwd-bwd, AE + A50/A75/A95)
- [x] Methodology gap closure (Middlebury R0.5/1/2 in report, Bad-10 catastrophic, s60+ bin, full Bad/A/AE per region, normalized Sintel→Middlebury, blur mask). H1 nuanced (clean win extended to s60+; final marginal-reverse), H10 quantified at 0.209 vs 0.328 normalized.
- [x] Critique-driven revision section (`§11 Critique-driven revisions`) folded in: GMFlow-refine three-way ablation, bootstrap CIs, F1 threshold sweep, blur–motion confound, alternative H10 normalization, quantization histograms, H9 downgrade.
- [x] AE↔EPE Pearson + Spearman per mask (Sintel) + per-sequence + global (Middlebury); s0-1 sub-pixel bin; per-pixel normalized EPE; fine-grained speed-bucket line plots (`epe/ae/nepe/bad1 vs speed`). H2 extended with s0-1 (RAFT 0.161 vs GMFlow 0.245) and nEPE@s1-3 (GMFlow 2× RAFT proportional error at slow motion).
- [x] **Critique-driven revision round** — fairness, statistical rigor, and contamination fixes:
  - GMFlow-with-refine as a third model column → §8 verdicts rewritten with both "vs basic" and "vs refine" reads. H2 / H3 / H4 falsified vs refine; H1 / H6 strengthened against capacity-matched comparison.
  - Paired sequence bootstrap → §3 main table's "RAFT clean EPE 1.446 < 1.484" headline is NOT significant (95% CI [−0.19, +0.20]). H1 final-pass "hair-margin" verdicts are NULL.
  - Boundary F1 threshold sweep → RAFT vs basic stable across τ ∈ {0.5, 1, 2}; refine wins at every τ.
  - Quantization histograms → "1/8-grid residual" claim softened to "coarse-grid pile-up at small k/8" (partially supported, basic shows it, refine reduces it).
  - Blur–motion confound → moderate (Pearson 0.64), not pure; `mountain_1` confirms blur mask catches genuine defocus too.
  - H10 alternative normalization → recomputed against Sintel `s0_10` denominator (matched motion regime); RAFT keeps directional win on both normalizers, magnitude depends on choice.
  - H9 retracted from "strongly supported" to "indicative, not verified" pending ≥16 GB-GPU re-run (2.0× / 2.5× rows contaminated by WSL2 unified-memory paging).
  - H7 stripped of "partial on matched-relative" qualifier (denominator artifact, not robustness signal).
- [ ] Final report (per project requirements — methodology + Phase 1 + H8 explicit "not pursued" + figures)
- [ ] Figures still to make: EPE-vs-iters curve from the full sweep, latency-vs-resolution log-log plot showing the GMFlow knee, per-sequence Clean→Final ΔEPE bar chart
