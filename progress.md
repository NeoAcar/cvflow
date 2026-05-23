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
- [x] `cvflow.metrics.sintel.SintelMetrics` — EPE+Bad-X over {all, matched, unmatched, s0-10, s10-40, s40+, disc, untex}
- [x] `cvflow.metrics.boundary_fscore.boundary_fscore` — F1 with `tol_px=2`
- [x] `cvflow.metrics.middlebury` — EE, AE, R0.5/R1.0/R2.0, A50/A75/A95 with `|flow|<1e9` mask

### Runs (Phase 1)
- [x] Sintel **clean** — RAFT (14 min) + GMFlow (6 min), full per-mask + saved `.npy`
- [x] Sintel **final** — RAFT (15 min) + GMFlow (8 min), full per-mask + saved `.npy`
- [x] Disc/Untex/F-score offline pass on all 4 (RAFT/GMFlow × clean/final)
- [x] Middlebury both models (8 pairs each, zero-shot Things-trained)
- [x] RAFT iter sweep `{4, 8, 12, 32}` on alley_1 + market_2
- [x] Latency + peak VRAM at 1024×436 (n=50, mixed sequences)
- [x] Photometric residual per-sequence on clean + final

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

### Open Phase 1 follow-ups (optional / not blocking)
- [ ] Report AE and A50/A75/A95 on Sintel (currently only on Middlebury)
- [ ] Native `motion_boundaries/` download from sintel.is.tue.mpg.de and re-compute Disc EPE + F-score against it (for direct comparability with papers using native mask)
- [ ] Full-dataset RAFT iter sweep (~50 min, ~28 GB) — currently only alley_1 + market_2
- [ ] Higher-resolution VRAM test to actually fire H9 (Sintel 1024×436 doesn't blow up; Spring 1920×1080 might)
- [ ] Per-pixel ΔEPE map visualization (Sintel Clean vs Final paired delta from §4 item 10) — currently only mean-per-sequence numbers
- [ ] Forward-backward consistency-based occlusion mask (methodology §2.2 cross-check) — currently using Sintel native only

### Writing
- [ ] Final report (per project requirements — methodology + Phase 1 + Phase 2 findings + figures)
- [ ] Figures: EPE-vs-iters curve, per-corruption heatmap (Phase 2), per-sequence bar chart, Clean→Final ΔEPE distribution
