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
- [x] `cvflow.metrics.sintel.SintelMetrics` — EPE + AE + Bad-1/3/5 + A50/A75/A95 over `{all, matched, unmatched, s0-10, s10-40, s40+, disc, untex}`
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

### Phase 2 (RobustSpring) — methodology §1.6, §4 items 11–12, hypothesis 8
- [ ] Download RobustSpring (spring-benchmark.org / Oei et al. 2026)
- [ ] `cvflow.datasets.robust_spring` — per-corruption iterator with SSIM-equalized severities
- [ ] `cvflow.metrics.robustness` — `R^c_EPE`, `R^c_1px`, `R^c_Fl` (Oei et al. 2026 Eq. 2)
- [ ] 0.05% subsampling per Oei et al. §3.2
- [ ] Aggregation: Average, Median, Schulze voting (Schulze 2018) over 20 corruptions
- [ ] Corruption × model heatmap
- [ ] Wall-clock budget ~1 hour with subsampling

### Phase 1 follow-up findings (all closed)
- AE and A50/A75/A95 added to `SintelMetrics`. RAFT A50=0.067 vs GMFlow A50=0.411 px; RAFT AE/all=3.91° vs GMFlow 5.78° on clean.
- Sintel native `motion_boundaries/` confirmed **not present** in `MPI-Sintel-training_extras.zip` and probably not publicly distributed. Derived Sobel+9×9 stays as the production path.
- Full-dataset RAFT iter sweep (all 23 seqs × {4, 8, 12, 32}): EPE 1.91 / 1.59 / 1.51 / 1.45 at 198 / 276 / 358 / 765 ms. **GMFlow (1.484 EPE @ 309 ms) Pareto-dominates RAFT-12**; only RAFT-32 beats GMFlow on accuracy.
- Resolution sweep — H9 fires at ≥2× Sintel. At 2560×1090: GMFlow 43.9 s/pair vs RAFT-32 6.9 s/pair (quadratic vs linear). VRAM grows similarly for both (~15 GB at 2.5×, paged via WSL2 unified memory).
- ΔEPE maps: 46 PNGs in `results/figures/delta_epe/<tag>/`. ambush_2 worst (RAFT +16 px, GMFlow +21 px median ΔEPE); bamboo_1 and sleeping_1 improve on Final due to motion blur.
- Fwd-bwd derived occlusion vs Sintel native: F1 ~0.60 / IoU ~0.43 for both models. GMFlow higher precision (0.752), RAFT higher recall (0.564). Methodology §2.2 cross-check satisfied.

### Writing
- [ ] Update `docs/phase1_results.md` to fold in the follow-up numbers (full-dataset iter sweep, resolution sweep, ΔEPE, fwd-bwd, AE+percentiles) — current file is the pre-follow-up snapshot
- [ ] Final report (per project requirements — methodology + Phase 1 + Phase 2 findings + figures)
- [ ] Figures still to make: EPE-vs-iters curve from the full sweep, latency-vs-resolution log-log plot showing the GMFlow knee, per-corruption heatmap (Phase 2), per-sequence Clean→Final ΔEPE bar chart
