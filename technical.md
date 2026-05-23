# Technical notes

## Repo tree

```
cvflow/
├── README.md                       project-level overview, env setup, run commands
├── CLAUDE.md                       behavioral guidelines + project-specific rules
├── progress.md                     done / pending checklist
├── technical.md                    this file
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
│   └── figures/                    (placeholder)
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
- `GMFlowWrapper(checkpoint, padding_factor=16, attn_splits=2, corr_radius=-1, prop_radius=-1, device=None)`:
  - Handles both `{'model': sd, ...}` and raw `sd` checkpoint formats.
  - `predict(img1_u8, img2_u8)` → normalize-inside-model GMFlow, returns `out['flow_preds'][-1]`.
  - Basic GMFlow config: `num_scales=1`, `feature_channels=128`, `attention_type='swin'`, `num_transformer_layers=6`, `ffn_dim_expansion=4`, `num_head=1`, `upsample_factor=8`.

### `masks/textureless.py`
`untext_mask(img_uint8, grad_thresh=5.0) -> bool[H,W]` — RGB→gray, Sobel(ksize=3), `‖∇I‖ < τ`, 3×3 dilate. Baker §4.3 convention.

### `masks/motion_boundary.py`
`disc_mask(gt_flow, grad_thresh=1.0) -> bool[H,W]` — Sobel on each component, `|∇u| + |∇v| > τ`, 9×9 dilate. Baker §4.3 convention.

### `masks/photometric.py`
`photometric_residual(img1, img2, gt_flow) -> float32[H,W]` — `cv2.remap` warps I₂ by GT flow, returns `|I₁ − I₂_warped|` mean over RGB.

### `metrics/sintel.py`
- `_Accum` — sum/count over masked EPE; tracks Bad-1/3/5 counts in one pass.
- `SintelMetrics`:
  - `.update(pred, gt, occlusion, invalid, seq, disc=None, untex=None)` — adds one pair across masks `{all, matched, unmatched, s0_10, s10_40, s40+, disc?, untex?}`. Per-seq + global.
  - `.global_summary()` → dict of `epe/<mask>` + `bad1/all`, `bad3/all`, `bad5/all`.
  - `.per_seq_epe_all()` → dict `{seq: epe_all}`.

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
Offline (CPU) evaluator. Reads saved `.npy` predictions + Sintel GT, computes Disc + Untex EPE + boundary F-score on top of the base masks. Adds two CLI knobs: `--disc-thresh`, `--untex-thresh`.

### `runners/run_middlebury.py`
Iterates 8 Middlebury GT pairs through both models, prints per-sequence `EE/AE/R0.5/R1.0/R2.0/A50/A95` + mean. Saves `.npy` to `results/predictions/<tag>/middlebury/<seq>/flow10.npy`.

### `runners/run_raft_itersweep.py`
RAFT iter sweep on a configurable sequence subset (default: alley_1 + market_2). For each iter level in `{4, 8, 12, 32}`: load fresh RaftWrapper, warm-up 3 frames, time per-pair with `torch.cuda.Event`, accumulate `SintelMetrics`. Output is one row per iter level.

### `runners/run_latency_vram.py`
Latency + peak VRAM at 1024×436. Loads a small set of mixed-sequence Sintel pairs, warm-up 5, time the next N (default 50). Reports median/mean/std/min/max latency in ms and `torch.cuda.max_memory_allocated` in MB. Runs RAFT first, then GMFlow after `empty_cache()`.

### `runners/run_photometric.py`
Iterates Sintel clean and final separately, computes per-pair `photometric_residual` masked by `(invalid==0) ∧ (occlusion==0)`, aggregates per-sequence means, prints clean / final / Δ table.

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
