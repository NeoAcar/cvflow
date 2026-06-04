# Technical notes

## Repo tree

```
cvflow/
├── README.md                       project-level overview, env setup, run commands
├── CLAUDE.md                       behavioral guidelines + project-specific rules
├── progress.md                     done / pending checklist
├── technical.md                    this file
├── src/cvflow/analysis/            NEW package for offline statistical analyses
│   └── bootstrap.py                paired sequence-level bootstrap (CI on Δmetric)
├── .venv/                          uv venv, Python 3.11, torch 2.6.0+cu124
├── datasets/
│   ├── Sintel/training/{clean,final,flow,occlusions,invalid,albedo,flow_viz}/
│   └── Middleburry/{other-data,other-gt-flow}/         (8 sequences with GT)
├── docs/
│   ├── eval_methodology.md         methodology spec (canonical)
│   ├── initial_plan.md             feasibility check + repo design + risks
│   ├── phase1_results.md           Phase 1 results + hypothesis verdicts
│   └── papers/{raft.pdf,gmflow.pdf}
├── RAFT/RAFT/                      upstream RAFT (princeton-vl/RAFT) + models/*.pth
├── gmflow/gmflow/                  upstream GMFlow (haofeixu/gmflow) + pretrained/*.pth
├── src/cvflow/                     our pipeline (see below)
├── results/
│   ├── predictions/<tag>/<dataset>/<pass>/<seq>/frame_NNNN.npy
│   ├── metrics/                    (placeholder — CSVs go here when needed)
│   └── figures/
│       ├── delta_epe/<tag>/{<seq>.png, _summary.csv}
│       └── speed_curves/{epe,ae,nepe,bad1}_vs_speed.png + speed_curves.csv
├── cache/masks/                    (placeholder — masks recomputed, not cached yet)
└── logs/                           per-run logs
```

`<tag>` is `raft-raft-things-iter32` or `gmflow-gmflow_things-e9887eda`.

## `src/cvflow/` — files and what they do

### `flow_io.py`
| Function | Purpose |
|---|---|
| `read_flo(path) -> float32[H,W,2]` | Middlebury `.flo` reader with `PIEH` magic check |
| `read_mask_png(path) -> uint8[H,W]` | Reads Sintel occlusion/invalid PNG, picks channel 0 if RGB |
| `read_image(path) -> uint8[H,W,3]` | RGB image reader via PIL |

### `datasets/sintel.py`
| Symbol | Purpose |
|---|---|
| `SintelPair` (dataclass) | `seq, idx, img1, img2, gt_flow, occlusion, invalid` |
| `Sintel(root, split='training', pass_='clean')` | Constructor; lists sequences alphabetically |
| `Sintel.pairs(seqs=None)` | Generator over pairs; `seqs=None` = all 23 |
| `Sintel.count()` | Total pairs (1041 on training) |

### `datasets/middlebury.py`
| Symbol | Purpose |
|---|---|
| `MiddleburyPair` (dataclass) | `seq, img1, img2, gt_flow` (single pair: frame10→frame11) |
| `Middlebury(root)` | Filters to sequences that have `other-gt-flow/<seq>/flow10.flo` |
| `Middlebury.pairs()` | Generator over the 8 GT'd sequences |

### `models/base.py`
`FlowModel` Protocol — `predict(img1_u8, img2_u8) -> float32[H,W,2]`, plus `.name` attribute. Both wrappers conform.

### `models/_padder.py`
`InputPadder(dims, mode='sintel', padding_factor=8)` — inlined `pad`/`unpad` with replicate padding. Replaces `from utils.utils import InputPadder` in both upstream repos. Avoids the `utils.utils` collision (see `CLAUDE.md`).

### `models/raft_wrapper.py`
- `_import_raft()` — clears any cached `utils*` modules, adds `RAFT/RAFT/core` to `sys.path`, returns the `RAFT` class.
- `RaftWrapper(checkpoint, iters=32, small=False, mixed_precision=False, device=None)`:
  - Strips `module.` prefix from checkpoint keys (DataParallel artifact).
  - `predict(img1_u8, img2_u8)` → normalize-inside-model RAFT, `padding_factor=8`, `mode='sintel'`, `test_mode=True`.
  - `.name` = `f"raft-{stem}-iter{iters}"`.

### `models/gmflow_wrapper.py`
- `_import_gmflow()` — clears `gmflow*` from `sys.modules`, adds `gmflow/gmflow` to `sys.path`, returns the `GMFlow` class.
- `GMFlowWrapper(checkpoint, padding_factor=16, attn_splits_list=(2,), corr_radius_list=(-1,), prop_radius_list=(-1,), num_scales=1, upsample_factor=8, device=None)`:
  - Handles both `{'model': sd, ...}` and raw `sd` checkpoint formats.
  - `predict(img1_u8, img2_u8)` → normalize-inside-model GMFlow, returns `out['flow_preds'][-1]`.
  - Defaults reproduce the basic (no-refinement) numbers identically. The **refine preset** (matching upstream `evaluate.sh`'s with-refinement block) is: `padding_factor=32, attn_splits_list=[2,8], corr_radius_list=[-1,4], prop_radius_list=[-1,1], num_scales=2, upsample_factor=4`. `run_sintel_eval.py` and `run_middlebury.py` expose this as `--gmflow-refine` so the six values flip atomically.

### `masks/textureless.py`
`untext_mask(img_uint8, grad_thresh=5.0) -> bool[H,W]` — RGB→gray, Sobel(ksize=3), `‖∇I‖ < τ`, 3×3 dilate. Baker §4.3 convention.

### `masks/motion_boundary.py`
`disc_mask(gt_flow, grad_thresh=1.0) -> bool[H,W]` — Sobel on each component, `|∇u| + |∇v| > τ`, 9×9 dilate. Baker §4.3 convention.

### `masks/photometric.py`
`photometric_residual(img1, img2, gt_flow) -> float32[H,W]` — `cv2.remap` warps I₂ by GT flow, returns `|I₁ − I₂_warped|` mean over RGB.

### `masks/blur.py`
`blur_mask(img_uint8, window=7, var_thresh=20.0) -> bool[H,W]` — windowed Laplacian variance via `cv2.boxFilter`, pixels below threshold = blurred/defocused. Methodology §2.6 / §2.8.

### `metrics/sintel.py`
- `_Accum` — sum/count over masked EPE; tracks Bad-1/3/5 counts in one pass.
- `_Accum`: stream-friendly per-mask accumulator. Tracks `sum_epe`, `sum_epe_sq` (for SD), `sum_ae`, `sum_n`, Bad-{1,3,5,10} counts (Bad-10 = catastrophic failure threshold), `sum_nepe` + `sum_nepe_n` (normalized EPE = `EPE/|gt|` on pixels with `|gt|>_NEPE_FLOOR=1.0`). Keeps two bounded paired reservoirs (`_SAMPLE_CAP = 5,000,000` masked-pixel values each) for percentile estimation and AE↔EPE correlation. Exposes `.epe()`, `.sd()`, `.ae()`, `.nepe()`, `.bad(t)`, `.percentile(q)`, `.correlation() → (pearson, spearman)`.
- `SintelMetrics`:
  - `.update(pred, gt, occlusion, invalid, seq, disc=None, untex=None, blur=None)` — adds one pair across masks `{all, matched, unmatched, s0_1, s0_10, s10_40, s40+, s60+, disc?, untex?, blur?}`. Internally computes AE via `cvflow.metrics.middlebury.angular_error_deg`. Per-seq + global.
  - `.global_summary()` → dict emitting per-mask `epe/<mask>`, `sd/<mask>`, `ae/<mask>`, `nepe/<mask>`, `bad{1,3,5,10}/<mask>`, `A{50,75,95}/<mask>`, `pearson/<mask>`, `spearman/<mask>` for every active mask.
  - `.per_seq_epe_all()` → dict `{seq: epe_all}`.
  - `.dump_per_seq(path)` — serialize per-sequence per-mask raw `(sum_epe, sum_epe_sq, sum_ae, sum_n, sum_bad{1,3,5,10}, sum_nepe, sum_nepe_n)` as JSON; the bootstrap module's input format.

### `analysis/bootstrap.py`
- `load_per_seq(path)` — load a `dump_per_seq` JSON.
- `paired_bootstrap(stats_a, stats_b, mask, metric, n_boot=10_000, seed=0) -> BootResult` — resample sequence indices with replacement (paired between A and B), recompute the pixel-weighted weighted mean per resample, return mean, point estimate Δ, 95% CI, two-sided p-value, and a `crosses_zero` flag. Supported `metric` values: `epe / ae / bad1 / bad3 / bad5 / bad10 / nepe`. Sequence-level (not pixel-level) bootstrap — pixels within a sequence are not independent.

### `metrics/boundary_fscore.py`
`boundary_fscore(pred_flow, gt_flow, grad_thresh=1.0, tol_px=2) -> (P, R, F1)` — builds Sobel-grad binary edge maps for both, then matches each map against the dilated other using a `2*tol_px+1` kernel.

### `metrics/middlebury.py`
- `endpoint_error(pred, gt)`, `angular_error_deg(pred, gt)`.
- `gt_valid_mask(gt)` — `|u|<1e9 ∧ |v|<1e9` (Middlebury invalid sentinel).
- `summary(pred, gt)` → `{ee_mean, ee_sd, ae_mean, ae_sd, R0.5, R1.0, R2.0, A50, A75, A95, valid_frac}`.

### `runners/step7_sanity.py`
The original gate runner. Runs RAFT and/or GMFlow over Sintel clean, prints `epe/all` against the §4 item 13 targets and ±10% tolerance. Used once for the planning gate; current production runner is `run_sintel_eval.py`.

### `runners/run_sintel_eval.py`
Production Sintel runner. CLI: `--model {raft,gmflow} --pass {clean,final} [--ckpt PATH] [--raft-iters N] [--no-save]`. Predicts, saves `.npy`, updates `SintelMetrics` over the 6 base masks, prints summary + per-seq EPE + ±10% pass/miss against GMFlow's published targets (for GMFlow only).

### `runners/eval_from_saved.py`
Offline (CPU) evaluator. Reads saved `.npy` predictions + Sintel GT, computes the full mask suite — `{all, matched, unmatched, s0_10, s10_40, s40+, s60+, disc, untex, blur}` — and for each emits EPE, SD, AE, Bad-{1,3,5,10}, A50/A75/A95 plus boundary F-score. CLI knobs: `--disc-thresh`, `--untex-thresh`, `--blur-window`, `--blur-thresh`. Runtime ~140 s for 1041 pairs on CPU.

### `runners/run_middlebury.py`
Iterates 8 Middlebury GT pairs through both models, prints per-sequence `EE/AE/R0.5/R1.0/R2.0/A50/A95` + mean. Saves `.npy` to `results/predictions/<tag>/middlebury/<seq>/flow10.npy`.

### `runners/run_raft_itersweep.py`
RAFT iter sweep over a configurable sequence subset (`--seqs`, default `alley_1 market_2`) and iter list (`--iters`, default `4 8 12 32`). For each iter level: load fresh RaftWrapper, warm-up 3 frames, time per-pair with `torch.cuda.Event`, accumulate `SintelMetrics`. Output is one row per iter level (EPE/all, s0_10, s10_40, s40+, median ms). Does not save predictions — only metrics. Full-dataset run (all 23 sequences) takes ~34 min on RTX 3050 Ti Laptop.

### `runners/run_latency_vram.py`
Latency + peak VRAM at 1024×436. Loads a small set of mixed-sequence Sintel pairs, warm-up 5, time the next N (default 50). Reports median/mean/std/min/max latency in ms and `torch.cuda.max_memory_allocated` in MB. Runs RAFT first, then GMFlow after `empty_cache()`.

### `runners/run_photometric.py`
Iterates Sintel clean and final separately, computes per-pair `photometric_residual` masked by `(invalid==0) ∧ (occlusion==0)`, aggregates per-sequence means, prints clean / final / Δ table.

### `runners/delta_epe_maps.py`
Reads saved `.npy` predictions on both passes, computes mean per-pixel `ΔEPE = EPE_final − EPE_clean` per sequence. Emits one PNG per sequence (diverging `RdBu_r` colormap centered at 0, clipped to 99th percentile of |Δ|) to `results/figures/delta_epe/<tag>/<seq>.png`, plus `_summary.csv` with mean / median / 95th-pct / fraction-worse-on-final / mean of positive / mean of negative ΔEPE.

### `runners/run_vram_resolution.py`
Latency + peak VRAM scan over upsample factors (default `1.0 1.5 2.0 2.5`). Upsamples Sintel pairs via `cv2.resize` (bilinear), runs `n` timed forward passes per factor for each model, catches `torch.cuda.OutOfMemoryError` and prints NaN if a config dies. Designed to fire methodology hypothesis 9 — confirmed: GMFlow latency grows quadratically, RAFT linearly above 1.5× Sintel.

### `runners/speed_curves.py`
Bins pixels by GT-flow magnitude on fine edges `[0,1,3,10,20,40,60,∞]`, accumulates EPE / AE / nEPE / Bad-1 per bucket per (model, pass), writes 4 PNG line plots and a `speed_curves.csv` to `results/figures/speed_curves/`. Same `nEPE` floor `|gt|>1` as `SintelMetrics`.

### `runners/middlebury_correlation.py`
Reads saved Middlebury `.npy` predictions, applies `gt_valid_mask`, computes per-sequence and global Pearson + Spearman between per-pixel EPE and AE for each model. No GPU.

### `runners/bootstrap_compare.py`
Loads two per-seq JSONs (from `eval_from_saved --dump-json`) and prints a CI'd Δ table over a configurable mask × metric grid. CLI: `--a <json> --b <json> [--masks ...] [--metrics ...] [--n-boot 10000] [--seed 0]`. Used in §11a for the critique-driven bootstrap comparisons.

### `runners/boundary_threshold_sweep.py`
Recomputes the §3 boundary-F1 metric at thresholds `{0.5, 1.0, 2.0}` (or any user-supplied list) for any set of model tags × {clean, final}, reading saved predictions. Output: PNG-less; just the F1 sensitivity CSV at `results/figures/boundary_threshold/sensitivity.csv` plus a pivot table on stdout. §11b consumer.

### `runners/quantization_check.py`
Histograms matched-pixel EPE on slow sequences (default `alley_2 + bandage_2`) and a fast control (default `ambush_4`) for any tag set. Visualizes alongside k/8 px reference lines; the summary CSV reports the fraction of matched pixels within ±0.02 of each `k/8` for `k=0..8`. §11f consumer.

### `runners/blur_motion_confound.py`
For each Sintel-clean sequence: per-pair `blur_mask` fraction (over valid pixels) and per-pair mean `|gt_flow|`. Reports Pearson + Spearman across the 23 sequences and writes a labeled scatter PNG. §11c consumer.

### `runners/run_fwdbwd_occlusion.py`
Forward-backward consistency check (Sundaram et al. 2010 / Meister et al. 2018):

    || f12(x) + f21(x + f12(x)) ||² > α · (||f12||² + ||f21(x+f12)||²) + β

Defaults `α=0.01`, `β=0.5`. Builds a derived occlusion mask per pair, compares against Sintel native `occlusion == 255` via precision/recall/F1/IoU on the valid pixels. Backward flow is computed by swapping `(img1, img2) → (img2, img1)`. Forward flow can be loaded from saved `.npy` cache via `--fwd-cache <path>` — RAFT-only with cached forwards is ~14 min, full RAFT+GMFlow with cached forwards ~22 min.

## Conventions

- **Flow tensor:** `float32[H,W,2]`, channel 0 = u (horizontal), channel 1 = v (vertical), positive u = rightward, positive v = downward. Frame1 → frame2 displacement in pixels.
- **Prediction storage:** `np.save` raw `.npy`, no compression. `~3.5 MB/pair` at 1024×436.
- **Sintel mask convention:** `invalid == 0` means valid GT; `occlusion == 0` means non-occluded; `occlusion == 255` means occluded.
- **Middlebury invalid sentinel:** `|u| ≥ 1e9 ∨ |v| ≥ 1e9` → discard. Must mask before computing EE/AE.
- **Sequence sort:** alphabetical via `sorted(...)` everywhere.

## Known gotchas

1. **`utils.utils` collision** — Solved with the inline `_padder.py`. Don't re-introduce `from utils.utils import InputPadder` in wrappers.
2. **RAFT checkpoint `module.` prefix** — `RaftWrapper.__init__` strips. Don't `nn.DataParallel`-wrap.
3. **GMFlow accepts dict-style `{'model': sd}` checkpoints** — wrapper handles both layouts.
4. **Sintel `motion_boundaries/` not on disk** — Disc mask is derived; thresholds are CLI flags.
5. **Middlebury path is misspelled `Middleburry`** — keep it that way to match the dataset's on-disk name.
6. **Python output buffering** — `python -m ... | tail` blocks until process exits; use `-u` for live progress (now used in long runs via `python -W ignore -u`).
7. **`-W ignore`** is set in long runs to suppress the upstream RAFT `torch.cuda.amp.autocast` deprecation warning, which floods stderr otherwise. Not load-bearing for correctness.

## Reproducibility caveats

- cuDNN kernel selection is not pinned, so two runs may diverge by ~1e-5 per pixel. Not material at EPE granularity (numbers stable to 4 decimal places across reruns on the same machine).
- The `weights_only=True` flag is used on `torch.load`. If we ever need optimizer state from a checkpoint it'd have to flip, but for inference-only it stays.
- Sintel global EPE on the same input is deterministic across the 4 wrapper instantiations tested (RAFT 12-iters, RAFT 32-iters, GMFlow basic, GMFlow with-refine).
