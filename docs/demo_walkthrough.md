# Demo walkthrough — code map + 4-minute video plan

Two things in one doc:
- **Part A — Code map**: where each piece lives, what it does, how the pieces compose.
- **Part B — 4-minute demo script**: minute-by-minute plan for the video, with the exact files/figures to show.

---

## Part A — Code map

### A.1 High-level pipeline

```
                        ┌─────────────────┐
   datasets/Sintel ──►  │ datasets/sintel │── (img1, img2, gt_flow, invalid, occlusion)
                        └────────┬────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
        models/raft_wrapper            models/gmflow_wrapper
        models/_padder                 (basic / refine presets)
                  │                             │
                  └──────────────┬──────────────┘
                                 ▼
                       float32 [H,W,2] prediction
                                 │
                                 ▼
                results/predictions/<tag>/<dataset>/<pass>/<seq>/frame_NNNN.npy
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   metrics/sintel.SintelMetrics          analysis / report runners
   metrics/middlebury                    (figures, bootstrap, sweeps)
   metrics/boundary_fscore
   masks/{motion_boundary,                          │
          textureless, blur,                        ▼
          photometric}                  docs/phase1_results.md
              │                         results/figures/...
              ▼
   results/per_seq_stats/*.json
   results/correlations/*.json
```

Three runtime stages:
1. **Predict** (GPU, ~6–16 min per `model × pass`): `runners/run_sintel_eval.py`, `runners/run_middlebury.py` → `.npy` files.
2. **Metric** (CPU, ~3 min per corner): `runners/eval_from_saved.py` → per-mask grid + per-sequence JSON.
3. **Analyze** (CPU): bootstrap, sweeps, figures.

### A.2 Where each piece lives

| Path | What it does |
|---|---|
| `src/cvflow/datasets/sintel.py` | Iterates `(img1, img2, gt_flow, invalid, occlusion, seq, idx)` pairs. 23 sequences × (frame_count − 1) = **1041 pairs** per pass. |
| `src/cvflow/datasets/middlebury.py` | 8 GT-bearing sequences from the "other" set. Applies the `|u|<1e9 ∧ |v|<1e9` invalid-flow filter. |
| `src/cvflow/models/_padder.py` | Inlined `InputPadder` — both upstream repos ship a `utils/utils.py`; collision avoided by inlining here. **Do not** re-import upstream. |
| `src/cvflow/models/raft_wrapper.py` | Loads `raft-things.pth` (strips `module.` DP prefix), runs 32 inference iters by default. |
| `src/cvflow/models/gmflow_wrapper.py` | Loads `gmflow_things` or `gmflow_with_refine_things`. Knobs: `attn_splits`, `corr_radius`, `prop_radius`, `num_scales`, `upsample_factor`, `padding_factor`. |
| `src/cvflow/masks/motion_boundary.py` | **Disc** mask — Sobel-L1 of GT flow `> 1.0`, 9×9 dilate. |
| `src/cvflow/masks/textureless.py` | **Untex** mask — Sobel-L2 of I₁ `< 5.0`, 3×3 dilate. |
| `src/cvflow/masks/blur.py` | **Blur** mask — local variance of 3×3 Laplacian in 7×7 box `< 20`. |
| `src/cvflow/masks/photometric.py` | Brightness-constancy residual `|I₁ − I₂(x+gt)|`. |
| `src/cvflow/metrics/boundary_fscore.py` | F1 between predicted-flow Disc mask and GT-flow Disc mask, 5×5 tolerance. |
| `src/cvflow/metrics/sintel.py` | `SintelMetrics` accumulator. Per-mask `_Accum` reservoirs (5M samples) → EPE / SD / AE / Bad-1,3,5,10 / nEPE / A50/A75/A95 / Pearson(EPE,AE) / Spearman(EPE,AE). Also dumps per-seq raw sums to JSON. |
| `src/cvflow/metrics/middlebury.py` | EE / AE / R0.5 / R1.0 / R2.0 per Baker et al. with the invalid sentinel filter. |
| `src/cvflow/analysis/bootstrap.py` | `paired_bootstrap(...)` — sequence-level 10k resamples, returns Δ + 95% CI + p. **Not pixel-level** (pixels in a scene aren't independent). |
| `src/cvflow/runners/run_sintel_eval.py` | Stage 1: predict & save Sintel `.npy`. CLI: `--model {raft,gmflow}` `--pass {clean,final}` `--gmflow-refine`. |
| `src/cvflow/runners/run_middlebury.py` | Stage 1 for Middlebury. CLI: `--model {raft,gmflow,both}` `--gmflow-refine`. |
| `src/cvflow/runners/eval_from_saved.py` | Stage 2: reads `.npy` predictions, applies masks, prints the §3b grid, optionally `--dump-json` per-seq stats. |
| `src/cvflow/runners/run_raft_itersweep.py` | RAFT @ iters ∈ {4, 8, 12, 32}, full Sintel clean. |
| `src/cvflow/runners/run_latency_vram.py` | n=50 forwards per model, median + p95 + `torch.cuda.max_memory_allocated`. |
| `src/cvflow/runners/run_vram_resolution.py` | §6b resolution sweep, factors 1.0× / 1.5× / 2.0× / 2.5× — H9 contaminated by WSL2 paging at 2.0/2.5×. |
| `src/cvflow/runners/run_photometric.py` | §7 brightness-constancy residual per sequence. |
| `src/cvflow/runners/run_fwdbwd_occlusion.py` | §7c — `‖f₁₂ + f₂₁(x+f₁₂)‖² > 0.01·‖·‖² + 0.5` vs Sintel native occlusion. |
| `src/cvflow/runners/delta_epe_maps.py` | §7b — per-pixel `EPE_final − EPE_clean` averaged per sequence, RdBu_r PNGs. |
| `src/cvflow/runners/speed_curves.py` | §7e — bins by `‖gt‖` into 7 buckets, plots EPE/AE/nEPE/Bad-1. |
| `src/cvflow/runners/blur_motion_confound.py` | §11c — per-sequence Pearson/Spearman between blur fraction and mean GT motion. |
| `src/cvflow/runners/boundary_threshold_sweep.py` | §11b — F1 at τ ∈ {0.5, 1.0, 2.0}. |
| `src/cvflow/runners/quantization_check.py` | §11f — histogram of matched-pixel EPE concentration near `k/8` multiples on slow vs fast sequence. |
| `src/cvflow/runners/dump_correlations.py` | §3b last two columns — Pearson + Spearman per mask, cached to `results/correlations/sintel_per_mask.json` (~13 min CPU). |
| `src/cvflow/runners/bootstrap_compare.py` | §11a — calls `paired_bootstrap` between two per-seq JSONs. |
| `src/cvflow/runners/make_report_figures.py` | 7 report-wide figures (pareto, region_bars_full, boundary sensitivity, Middlebury per-seq, clean→final delta, AE/EPE correlation grids). |
| `src/cvflow/runners/make_paper_figures.py` | Per-sample 3-model qualitative panels (input / GT / 3 flows / EPE / diff / regions). |
| `src/cvflow/runners/step7_sanity.py` | The pre-flight gate against Teed&Deng / GMFlow published numbers. |

### A.3 What's in `results/` (after a full run)

```
results/
├── predictions/                           # raw .npy flow (~3.5 GB per (model,pass))
│   ├── raft-raft-things-iter32/sintel/{clean,final}/<seq>/frame_NNNN.npy
│   ├── gmflow-gmflow_things-e9887eda/...
│   └── gmflow-gmflow_with_refine_things-36579974/...
├── per_seq_stats/                         # JSON, weighted-mean ready
│   └── {raft,gmflow_basic,gmflow_refine}_{clean,final}.json
├── correlations/sintel_per_mask.json      # Pearson/Spearman per mask (3b)
├── bootstrap/*.txt                        # 6 paired-bootstrap reports
└── figures/
    ├── report/        # paper-ready: pareto, region bars, F1 sweep, …
    ├── paper/         # per-sample qualitative panels
    ├── speed_curves/  # EPE/AE/nEPE/Bad-1 vs |gt|
    ├── delta_epe/     # per-sequence ΔEPE PNGs (RdBu_r)
    ├── boundary_threshold/sensitivity.csv
    └── quantization/  # k/8-grid histograms
```

### A.4 Configuration touch points

| What | Where | Default |
|---|---|---|
| RAFT iters | `models/raft_wrapper.py:__init__(iters=)` | 32 |
| GMFlow basic preset | `models/gmflow_wrapper.py` | `attn_splits=[2], corr_radius=[-1], prop_radius=[-1], padding_factor=16` |
| GMFlow refine preset | `models/gmflow_wrapper.py` (via `--gmflow-refine` in runners) | `num_scales=2, padding_factor=32, attn_splits=[2,8], corr_radius=[-1,4], prop_radius=[-1,1], upsample_factor=4` |
| Disc threshold + dilate | `masks/motion_boundary.py` / `eval_from_saved.py --disc-thresh` | `>1.0`, 9×9 |
| Untex threshold + dilate | `masks/textureless.py` / `--untex-thresh` | `<5.0`, 3×3 |
| Blur window + threshold | `masks/blur.py` / `--blur-window --blur-thresh` | window 7, var `<20` |
| Boundary F1 tolerance | `metrics/boundary_fscore.py` | `tol_px=2` → 5×5 band |
| Bootstrap n_boot | `analysis/bootstrap.py` | 10,000 |

---

## Part B — 4-minute demo script

**Goal:** show *what we built*, *what it produces*, and the *one headline finding*. Not a paper read-through. Camera shares the screen; you narrate over a fixed sequence of files/figures.

### Setup before recording (1 min — not on tape)

1. Repo open in the IDE at root.
2. Terminal with venv active and `PYTHONPATH=src`.
3. Two tabs pre-opened:
   - `docs/phase1_results.md`
   - `results/figures/report/pareto.png`
4. A small test invocation already cached so you can show a runner finishing quickly (use one short sequence like `alley_1`).

### Minute 1 — *What this is* (0:00–1:00)

- **0:00–0:10** Title card / opening sentence:
  > "Topic E final project — inference-only comparison of RAFT and GMFlow on Sintel and Middlebury. Three checkpoints, all Things-trained, zero-shot."
- **0:10–0:30** Open `README.md`, scroll to "Layout" — point at the three regions:
  - `src/cvflow/` (our pipeline)
  - `RAFT/`, `gmflow/` (upstream model code, untouched)
  - `datasets/`, `results/`, `docs/`
- **0:30–1:00** Open `docs/demo_walkthrough.md` §A.1 — show the **pipeline diagram**:
  > "Three stages: predict → metric → analyze. Predictions are saved as `.npy`, so every analysis after that is re-runnable on CPU without touching the GPU."

### Minute 2 — *The pipeline, code-first* (1:00–2:00)

- **1:00–1:20** Open `src/cvflow/runners/run_sintel_eval.py`:
  > "Stage 1. One call: load model wrapper, iterate Sintel pairs, save `[H,W,2] float32` to `.npy`."
- **1:20–1:35** Open `src/cvflow/models/gmflow_wrapper.py` and show `predict(...)`:
  > "Both wrappers share a single contract: `predict(img1_u8, img2_u8) → float32[H,W,2]`. Same call for RAFT, GMFlow-basic, and GMFlow-refine."
- **1:35–1:50** Open `src/cvflow/metrics/sintel.py` `SintelMetrics`:
  > "Stage 2. Per-mask accumulator. For every pair we update EPE, AE, Bad-1/3/5/10, and a 5M-sample reservoir for percentiles and Pearson/Spearman."
- **1:50–2:00** Open `src/cvflow/masks/blur.py`:
  > "Sintel doesn't ship a blur mask. We derive it: 3×3 Laplacian, local variance in a 7×7 box, threshold at 20. Same pattern for Disc, Untex, Boundary F1 — all documented in `docs/derived_masks.md`."

### Minute 3 — *The result, one headline* (2:00–3:00)

- **2:00–2:15** Open `results/figures/report/pareto.png`:
  > "This is our headline. Latency on x, Sintel-clean EPE on y. RAFT iter sweep on the curve, three n=50 cross-model markers."
- **2:15–2:45** Talk through three points on the figure:
  > "GMFlow-basic strictly dominates RAFT-12 — same accuracy, ~100 ms cheaper. RAFT-32 sits on the Pareto frontier with 58 ms of slack to GMFlow-refine. **GMFlow-refine is 26% more accurate than RAFT-32 for only 10% more latency** — the trade slope strongly favors refine."
- **2:45–3:00** Open `docs/phase1_results.md` and scroll to §11a bootstrap table:
  > "We don't just point-estimate. Every close call has a sequence-level paired bootstrap, 10k resamples — half the original 'RAFT wins' headlines turned NULL once we tested against the 23-sequence variance."

### Minute 4 — *Run it, wrap up* (3:00–4:00)

- **3:00–3:25** Terminal — run a tiny, fast command for a "it really works" beat:
  ```bash
  python -m cvflow.runners.eval_from_saved \
    --pred-root results/predictions/raft-raft-things-iter32 \
    --pass clean --seqs alley_1 market_2
  ```
  Show the `EPE / matched / unmatched / Bad-1 / boundary F1` table appearing in <30 s.
- **3:25–3:45** Open `results/figures/report/region_bars_full.png` and `results/figures/report/clean_to_final_delta.png` — quick flash, one sentence each:
  > "Per-region EPE bars, clean and final. Per-sequence Clean→Final degradation — `ambush_2` dominates the budget for every model."
- **3:45–4:00** Close shot, three-bullet wrap:
  > "Three checkpoints, full Sintel + Middlebury, 10k-resample bootstrap on every verdict, reproducible from `.npy`. Headline: refine beats RAFT-32 by 26% accuracy at 10% more latency. Details in `docs/phase1_results.md`."

### Cheat sheet (talking points in order)

| Time | What's on screen | One-line script |
|---|---|---|
| 0:00 | Title slide | "Inference-only RAFT vs GMFlow on Sintel + Middlebury, zero-shot." |
| 0:15 | `README.md` Layout | "Our code under `src/cvflow`, upstream models untouched." |
| 0:30 | `demo_walkthrough.md` diagram | "Predict → Metric → Analyze, with `.npy` checkpoints between stages." |
| 1:00 | `run_sintel_eval.py` | "Stage 1 saves predictions as raw `.npy`." |
| 1:20 | `gmflow_wrapper.py` `predict()` | "Single `predict()` contract for all three models." |
| 1:35 | `metrics/sintel.py` | "Per-mask accumulator + reservoir for percentiles and correlation." |
| 1:50 | `masks/blur.py` | "Five derived masks; kernels and thresholds in `docs/derived_masks.md`." |
| 2:00 | `report/pareto.png` | "Headline figure." |
| 2:15 | Same figure | "Refine is 26% better than RAFT-32 at 10% more latency." |
| 2:45 | `phase1_results.md §11a` | "Bootstrap turns half the close calls NULL — we don't trust point estimates alone." |
| 3:00 | Terminal | "Reproduce a slice in 30 seconds from cached predictions." |
| 3:25 | `region_bars_full.png` + `clean_to_final_delta.png` | "Per-region and per-sequence pictures of the same story." |
| 3:45 | Wrap shot | "Three models, full Sintel+Middlebury, every verdict bootstrap-tested." |

### Recording notes

- Keep cursor moving — viewers track it; idle cursor on a slide = lost attention.
- For terminal beat: pre-run the command once so the relevant Python imports are warm; otherwise the `import torch` pause eats 4 seconds of your minute.
- If you have to cut something, drop the §11a bootstrap mention (2:45–3:00) — the headline plus a code-level pipeline tour is the minimum viable demo.
- Don't show `rapor.tex` or the methodology PDF in the demo — those are for the written submission, not the video.
