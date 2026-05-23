# Initial Plan — Topic E (RAFT vs GMFlow, inference-only)

Planning session output. No code under `src/` yet. Approval gate at the end.

Two discrepancies surfaced before starting:

1. Methodology file is `docs/eval_methodology.md`, not `docs/optical_flow_evaluation_methodology.md`.
2. The brief refers to "Section 4 item 13" of the methodology, but Section 4 currently lists only items 1–9. I treat the RAFT 1.48 / GMFlow 0.94 ε_clean reference as a hard sanity-check target (see §3 step 7) and flag the numbering mismatch in §5.

---

## 1. Feasibility check

### 1.1 Sintel training split — present and (mostly) complete

Direct disk inspection of `datasets/Sintel/training/`:

```
albedo/  clean/  final/  flow/  flow_viz/  invalid/  occlusions/
README.txt
```

- **23 sequences** per pass (alley_1, alley_2, ambush_2/4/5/6/7, bamboo_1/2, bandage_1/2, cave_2/4, market_2/5/6, mountain_1, shaman_2/3, sleeping_1/2, temple_2/3).
- **1064 frames per pass** (clean and final each), **1041 GT flow files** (23 seqs × (frame_count − 1)), 1041 occlusion masks, 1064 invalid masks. Counts are internally consistent: there is exactly one flow/occlusion per pair, one invalid per frame.
- Frame resolution **1024×436**, 8-bit RGB PNG. Occlusion and invalid masks are 1024×436 8-bit grayscale.
- A sample `.flo` opens with magic `b'PIEH'`, header reports W=1024 H=436, file size 3,571,724 B = `1024·436·2·4 + 12` — standard Middlebury format, no corruption.
- `albedo/` and `flow_viz/` are present but not needed.

Methodology Section 2 masks vs disk reality:

| Section | Mask | On disk? | Plan |
|---|---|---|---|
| 2.1 Textureless | None native | derived (structure tensor) | derive, cache |
| 2.2 Occlusions | `occlusions/` | **yes** | use native |
| 2.3 Large displacement | derived from GT | n/a | compute from `‖gt_flow‖` |
| 2.4 Motion boundaries | `motion_boundaries/` | **NO — missing from download** | **derived Sobel on GT flow (methodology's fallback path)** |
| 2.5 Photometric residual | derived (warp + diff) | n/a | derive on demand |
| 2.6 Motion blur | derived (Laplacian var) | n/a | derive on demand |
| invalid | `invalid/` | **yes** | apply before EPE |

The only missing GT mask is `motion_boundaries/`. The Sintel-complete tarball does not include it by default — it is a separate optional download on `sintel.is.tue.mpg.de`. The methodology already lists a derived Sobel/dilate alternative (§2.4) and I commit to that to avoid an extra download and to keep the F-score self-consistent (derive boundary on both pred and GT the same way).

### 1.2 Middlebury "other" — partial GT

```
other-data/   12 sequences
other-gt-flow/ 8 sequences
```

- 12 sequences have image frames. **Only 8 have GT flow**: Dimetrodon, Grove2, Grove3, Hydrangea, RubberWhale, Urban2, Urban3, Venus. Beanbags, DogDance, MiniCooper, Walking have **no** GT — cannot be used for accuracy metrics. (They could be used for visual / VRAM-latency only, which the methodology does not call for.)
- Each GT'd sequence provides exactly **one pair** (`frame10 → frame11`, GT `flow10.flo`). Total: **8 evaluable pairs**.
- Resolutions vary across sequences: Grove2/Urban2 are 640×480, Dimetrodon is 584×388, etc. Both wrappers' `InputPadder` handles variable resolution.
- `.flo` files open correctly (`PIEH` magic on all four spot-checked sequences).

### 1.3 RAFT repo — readable, no checkpoints

- `RAFT/RAFT/core/raft.py:86` — `forward(image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False)`.
- Inputs: `[B, 3, H, W]` float, pixel values in `[0, 255]`. Normalization `2·(I/255) − 1` is **inside** the model.
- Padding: `RAFT/RAFT/core/utils/utils.py:7` `InputPadder(dims, mode='sintel')` → pads to multiples of 8 with symmetric replicate padding for Sintel mode.
- `test_mode=True` returns `(flow_low, flow_up)`; `flow_up` is the upsampled prediction matching input H×W.
- Default inference iters per the proposal is 32; the in-tree `demo.py` uses 20; training default 12. Headline will be **32 iters**.
- Checkpoint: download via `RAFT/RAFT/download_models.sh` (Dropbox zip → `raft-things.pth`, `raft-sintel.pth`, `raft-kitti.pth`, `raft-chairs.pth`, `raft-small.pth`). Headline checkpoint **`raft-things.pth`** (~21 MB) — clean zero-shot to Sintel, matching the proposal's generalization framing.
- Quirk: `demo.py` wraps the model in `nn.DataParallel` before `load_state_dict`. Checkpoint keys are prefixed with `module.`. Wrapper must either reproduce the `DataParallel` wrap or strip the prefix.
- Quirk: `RAFT/RAFT/core/` uses **absolute** intra-package imports (`from update import ...`). Loading via `sys.path.insert(0, "RAFT/RAFT/core")` is the upstream-sanctioned pattern (see `demo.py:1-2`).

### 1.4 GMFlow repo — readable, no checkpoints

- `gmflow/gmflow/gmflow/gmflow.py:92` — `forward(img0, img1, attn_splits_list, corr_radius_list, prop_radius_list, pred_bidir_flow=False)` → returns `dict` with `'flow_preds'` (list); the final element is the prediction at full resolution.
- Inputs: `[B, 3, H, W]` in `[0, 255]`. Normalization is **inside** the model (`utils.normalize_img`: divide by 255 then ImageNet mean/std).
- Padding: `gmflow/gmflow/utils/utils.py:5` `InputPadder(..., padding_factor=8)`. The `main.py` default for evaluation is `padding_factor=16`; the **with-refinement** config uses `padding_factor=32`. We use the basic model → `padding_factor=16`.
- Required hyperparameters for the basic Things-trained model (from `scripts/evaluate.sh`): `attn_splits_list=[2]`, `corr_radius_list=[-1]`, `prop_radius_list=[-1]`, `num_scales=1`, `upsample_factor=8`. These are the GMFlow paper's headline (no-refinement) settings.
- Checkpoint: from Google Drive (link in README). Headline checkpoint **`gmflow_things-e9887eda.pth`**. Reference numbers baked into `scripts/evaluate.sh`: Sintel clean EPE = 1.495, final EPE = 2.955, with full speed-bucket and Bad-X breakdowns — invaluable as a sanity target.
- Imports are relative within `gmflow/gmflow/gmflow/`; importing from `gmflow/gmflow/` as a package root is clean.

### 1.5 Compute / environment

- `nvidia-smi` fails (`NVML: N/A`), but `/dev/dxg` is present — typical WSL2 state where the DirectX-GPU passthrough is wired but the NVML userspace tool isn't installed. `torch.cuda.is_available()` is the real test; it has to be run after we install torch (Python in the shell currently has no torch installed). **Risk, not a blocker** — A100 on Colab is the fallback per the brief.
- `uv 0.9.0` and `conda 26.1.1` both available.

### 1.6 RobustSpring

Not on disk. Marked **Phase 2** per the brief. The plan below assumes Phase 1 = Sintel + Middlebury only. Methodology items that hinge on RobustSpring (Section 2.7, hypothesis 8, suggested-reporting item 5) are explicitly deferred.

### 1.7 Verdict

**Feasible right now (Phase 1):**

- All metrics in methodology §1 (EPE, Bad-X, speed-buckets, matched/unmatched, EPE@boundaries with derived mask, boundary F-score, catastrophic %, latency, VRAM).
- Failure-mode masks §2.1 (derived), §2.2 (native), §2.3 (derived), §2.4 (derived only), §2.5 (derived), §2.6 (derived).
- All hypotheses 1–7, 9, 10 in §3.
- Must-measure items 1–9 in methodology §4.

**Blocked / deferred:**

- Methodology §2.7 (noise/weather), §2.8 (defocus on Final is qualitative-only), hypothesis 8 (corruption sensitivity), suggested-reporting item 5 — all require **RobustSpring → Phase 2**.
- Methodology §2.4 native motion-boundary mask is unavailable; **derived path only** in Phase 1.

**Open clarification (not blocking):** the "Section 4 item 13 — Oei et al. 2026 ε_clean (RAFT 1.48, GMFlow 0.94)" reference. The 1.48 number is consistent with `raft-things.pth` on Sintel clean. The 0.94 number is **not** consistent with any single basic GMFlow checkpoint I can identify — `gmflow_things-e9887eda.pth` on Sintel clean is 1.495 (their `scripts/evaluate.sh` is explicit), and 0.94 sits between the basic and the with-refinement results. Possibilities: Oei et al. used `gmflow_with_refine_things-36579974.pth` (clean EPE 1.084 — still not 0.94), the Sintel-finetuned `gmflow_sintel-0c07dcb3.pth` on a different split, or a different metric definition. I will flag this in §5 and proceed with the GMFlow paper's own published number (1.495) as the primary checkpoint-sanity target; the Oei 0.94 figure stands as a secondary cross-reference once the source is verified.

---

## 2. Repo layout

Single proposal.

```
src/cvflow/
    flow_io.py              # .flo read (PIEH magic) → np.float32[H,W,2]; PNG mask read
    datasets/
        sintel.py           # iterate (img1, img2, gt_flow, occ_mask, invalid_mask, seq, idx)
        middlebury.py       # iterate 8 GT'd pairs only
    models/
        base.py             # Protocol: predict(img1_u8, img2_u8) -> np.float32[H,W,2]
        raft_wrapper.py     # owns: sys.path patch, InputPadder(mode='sintel'), DataParallel-prefix strip, iters arg
        gmflow_wrapper.py   # owns: InputPadder(padding_factor=16), attn_splits/corr_radius/prop_radius lists
    masks/
        textureless.py      # structure-tensor on frame1; cached
        motion_boundary.py  # Sobel on GT flow magnitude + dilate(3); cached
        speed.py            # s0-10, s10-40, s40+ derived from GT flow at metric time
        photometric.py      # warp frame1 with GT flow, |I1 - I2_warped|
    metrics/
        epe.py              # EPE under arbitrary boolean mask; applies invalid automatically
        bad_x.py            # Bad-1/3/5 px outlier rate
        boundary_fscore.py  # F-score between pred and GT boundary masks
        catastrophic.py     # % pixels with EPE > 10
        latency_vram.py     # torch.cuda.Event + max_memory_allocated
    runners/
        run_inference.py    # iterate dataset × model → save .npy predictions
        run_metrics.py      # iterate predictions → CSV
        run_iter_sweep.py   # RAFT iter sweep (limited sequences)
        run_latency_vram.py
results/
    predictions/<model_tag>/<dataset>/<pass>/<seq>/frame_NNNN.npy
    metrics/<model_tag>_<dataset>_<pass>.csv
    figures/
cache/
    masks/<dataset>/<seq>/<frame_NNNN>_{textureless,motion_boundary}.npy
logs/<timestamp>_<model_tag>_<dataset>.log
```

**Unified `Model` interface, two wrappers**. Justification: the per-model differences (RAFT's `iters` argument, in-model normalization vs in-utility normalization, `padding_factor=8` vs `16`, `(flow_low, flow_up)` tuple vs `{'flow_preds': [...]}` dict, `DataParallel` prefix on the RAFT checkpoint) all live below the `predict(img1_u8, img2_u8) -> H×W×2 np.float32` contract. Runner code becomes model-agnostic; per-model peculiarities don't leak into metric and dataset code. A single class with branches would couple the two repos' quirks into one body — strictly worse for testability.

**Predictions stored as per-(model, dataset, pass, sequence) folders of `.npy` files**. Disk math: Sintel pair at float32 = `1024·436·2·4 ≈ 3.5 MB`. 1041 pairs × 3.5 MB ≈ **3.6 GB** per (model, pass). Four combinations of {RAFT, GMFlow}×{clean, final}: **~14 GB**. RAFT iter sweep `{4, 8, 12, 32}` × 2 passes adds another ~28 GB if run full-dataset — but per the methodology only the latency-accuracy curve is required, so the sweep runs on a **2-sequence representative subset** (`alley_1`, `market_2`) → adds ~0.3 GB. Middlebury is negligible. **Total ~15 GB.** HDF5 is rejected: small file count, debuggability (`np.load` is universal, no h5py dependency), and per-frame parallelism friendliness.

**Mask caching policy**:

- Textureless (structure tensor on input image): cache. Cost is non-trivial (~10–50 ms per frame) and the mask is reused by every metric pass.
- Derived motion-boundary (Sobel on GT flow + dilate): cache. Cheap, but caching enforces determinism across re-runs.
- Speed-bucket masks: **recompute per metric call**. Cost is one `np.linalg.norm` on the GT flow — microseconds. No caching gain.
- Native Sintel `occlusions/` and `invalid/`: read from disk each time, no caching layer — they're already on disk in usable form.
- Photometric residual: compute on demand, not cached. Only used for §2.5 analysis, not on the metric hot path.

---

## 3. Implementation order

Each step has a verification check. The plan only advances when the check passes.

1. **Env bootstrap**: `uv venv .venv`, install torch (CUDA-12 wheel), numpy, opencv-python, pillow, scipy, einops (GMFlow uses it). Skip both repos' own `environment.yml` / conda recipes — they pin PyTorch 1.6/1.9, which is incompatible with modern GPU drivers and offers no integration benefit. **Verify**: `python -c "import torch; print(torch.cuda.is_available())"` returns `True`. If `False`, switch to Colab A100 (per brief) and re-run this same check there.
2. **`flow_io.py`**. **Verify**: load `datasets/Sintel/training/flow/alley_1/frame_0001.flo` → shape `(436, 1024, 2)`, dtype `float32`, mean magnitude ~0–2 px (alley_1 is a slow sequence). Load `datasets/Middleburry/other-gt-flow/Grove2/flow10.flo` → shape `(480, 640, 2)`.
3. **Sintel dataset iterator**. **Verify**: iterate clean pass → expect exactly 1041 tuples; every tuple's GT/occlusion/invalid files exist; sample mid-sequence pair displays sensible images.
4. **Middlebury dataset iterator**. **Verify**: yields exactly 8 tuples (one per GT'd sequence).
5. **RAFT wrapper + checkpoint download**. Run `RAFT/RAFT/download_models.sh` (or fetch only `raft-things.pth`). **Verify**: file size matches Dropbox's; `predict` on `(alley_1/frame_0001, alley_1/frame_0002)` returns shape `(436, 1024, 2)` with finite values; EPE against `frame_0001.flo` GT under 1.0 (alley_1 is easy).
6. **GMFlow wrapper + checkpoint download** (`gmflow_things-e9887eda.pth`). **Verify**: same pair, finite output, EPE < 1.0.
7. **Headline checkpoint sanity (this is where the Oei reference lands)**. Run both models over full Sintel clean (1041 pairs) with default iters/configs (RAFT 32 iters, GMFlow `attn_splits=[2] corr_radius=[-1] prop_radius=[-1]`), compute global EPE with invalid mask applied. **Verify**:
    - GMFlow ≈ **1.495 ± 0.02** (target from `gmflow/gmflow/scripts/evaluate.sh:8`).
    - RAFT ≈ **1.43 ± 0.05** (RAFT paper, `raft-things` on Sintel-train clean; 1.48 also acceptable — methodology cites Oei 1.48).
    - GMFlow 0.94 cross-reference (Oei 2026) is **not expected to match this checkpoint** — flagged separately in §5.
    - If either number deviates by more than the tolerance: stop and debug. Likely culprits are checkpoint key mismatch, wrong padding mode, wrong normalization, or invalid-mask handling.
8. **Per-mask EPE: speed buckets + matched/unmatched + Bad-X**. **Verify** GMFlow: `s0_10=0.457, s10_40=1.770, s40+=8.257, 1px=0.161, 3px=0.059, 5px=0.040` ± a few %. Numbers come from the same `evaluate.sh` comment block. If GMFlow matches the published per-bucket numbers, our mask logic and bucket boundaries are correct.
9. **Derived textureless and motion-boundary masks**. **Verify**: visualize on `cave_4` (textureless cave walls — should light up) and `market_2` (high-motion — boundaries should trace moving figures).
10. **Boundary F-score metric**. **Verify**: F-score of GT-against-itself = 1.0 (sanity); F-score of zero-flow against `alley_1` GT ≈ 0 (sanity).
11. **Sintel final pass for both models**. **Verify** GMFlow final EPE ≈ **2.955 ± 0.05** (`evaluate.sh:13`).
12. **Middlebury cross-dataset (both models, same things-trained checkpoints)**. **Verify**: numbers are finite, plausibly in the range 0.2–2 px; per-sequence breakdown produced.
13. **RAFT iter sweep on {`alley_1`, `market_2`} × {4, 8, 12, 32}**. **Verify**: EPE monotonically decreases with iters (or near-monotone); produces the EPE-vs-latency plot required by hypothesis 6 / must-measure item 6.
14. **Latency + peak VRAM at 1024×436** for both models, headline configs. Use `torch.cuda.Event` + `torch.cuda.max_memory_allocated`; warm up 5 pairs, time over 50. **Verify**: GMFlow latency in the ballpark of the paper's 57 ms (V100) — on consumer/A100 expect ~26–60 ms.
15. **Clean-vs-Final paired delta, photometric residual analysis**. **Verify**: residual is small (<5 in 0–255 intensity units) on `alley_1` clean pairs, larger on `market_2` final pairs.

After step 15, all must-measure items 1–9 from methodology §4 are covered. Hypotheses 1–7, 9, 10 are evaluable. Hypothesis 8 (corruption sensitivity) waits for RobustSpring in Phase 2.

---

## 4. Dataset reference

| Dataset / split | Path | Pairs evaluable | Resolution | Masks on disk | Gotchas |
|---|---|---|---|---|---|
| Sintel train Clean | `datasets/Sintel/training/clean` | 1041 (23 seqs) | 1024×436 | `occlusions/`, `invalid/` | apply `invalid` (255 = unreliable) before EPE; `occlusions/` is 0/255 (255 = occluded) — confirm convention by reading one pixel against Sintel's `bundler/README` if there's any doubt at step 8 |
| Sintel train Final | `datasets/Sintel/training/final` | 1041 (23 seqs) | 1024×436 | same `occlusions/`, `invalid/` | Final adds atmospheric/blur/grain; pair indexing identical to Clean |
| Middlebury other | `datasets/Middleburry/other-data` + `other-gt-flow` | **8** (Dimetrodon, Grove2, Grove3, Hydrangea, RubberWhale, Urban2, Urban3, Venus) | 584×388 to 640×480 | none provided | One pair per seq (`frame10`→`frame11`, GT `flow10.flo`); 4 seqs (Beanbags, DogDance, MiniCooper, Walking) lack GT and are unused; `.flo` is PIEH-magic same as Sintel |
| RobustSpring | not downloaded | — | — | — | **Phase 2** |

Notes:

- `.flo` format is Middlebury-standard: `b'PIEH'` magic + `int32` W + `int32` H + `float32[H,W,2]` data. Confirmed for Sintel and Middlebury samples.
- Sintel `albedo/` and `flow_viz/` are present but unused.
- A `flow_code/` directory ships with Sintel — MATLAB and C reference readers; we don't use it (we have our own Python reader), but it's useful to grep when in doubt about mask conventions.

---

## 5. Risks

1. **`motion_boundaries/` missing.** Methodology §2.4 lists a native mask we don't have. We commit to the derived Sobel-on-GT-flow + dilate path (which §2.4 already describes as the alternative). Boundary F-score becomes "derived-vs-derived" — internally consistent but not directly comparable to papers that use the native mask. Note this in the methodology section of the final report.
2. **WSL CUDA not yet verified.** `nvidia-smi` fails; `/dev/dxg` is present. Won't know whether `torch.cuda.is_available()` returns True until step 1 of the implementation. Fallback per brief: A100 on Colab. No CPU fallback path.
3. **GMFlow VRAM at high resolution.** Transformer attention is O(N²). Sintel 1024×436 is the resolution the paper benchmarks (≈26 ms on A100, ~60 ms on V100) and is fine. Middlebury maxes at 640×480, smaller. If a later high-res experiment is added, expect a blow-up; default to Colab A100 in that case.
4. **Checkpoint-sanity number ambiguity.** "RAFT 1.48 / GMFlow 0.94 (Oei et al. 2026)" doesn't match a single basic GMFlow checkpoint we can identify (`gmflow_things` → 1.495, `gmflow_with_refine_things` → 1.084). Possibilities: Oei used the with-refinement variant on a different split, the Sintel-finetuned checkpoint, or a metric variant. **Concrete request before locking the report**: point me at the Oei paper / table so we can pin down which checkpoint and metric produces 0.94. Until then, the GMFlow paper's own published 1.495 on `gmflow_things` is the primary sanity target.
5. **Headline checkpoint choice.** Committed: `raft-things.pth` and `gmflow_things-e9887eda.pth` (Things-trained, never seen Sintel or Middlebury). Rationale: matches the proposal's zero-shot framing; gives both models an equal cross-dataset footing for hypothesis 10 (Sintel → Middlebury). Sintel-finetuned variants (`raft-sintel.pth`, `gmflow_sintel-0c07dcb3.pth`) would inflate the Sintel-domain numbers and contaminate the Sintel-to-Middlebury generalization claim. If you want headline numbers that match in-domain finetuned state-of-the-art instead, this decision needs to flip — flag in review.
6. **RAFT iter count.** Multiple defaults in the upstream code: 12 (training), 20 (demo.py), 32 (paper inference). Committed: **32 iters** for the headline, full sweep `{4, 8, 12, 32}` for the iter-sweep run on the 2-sequence subset. State this number explicitly in every reported RAFT result.
7. **Mixed precision off.** RAFT supports `--mixed_precision`; we leave it off for the headline. Mention in methodology to avoid grader confusion if a reader compares against RAFT paper numbers (which use FP32 inference).
8. **Env decision.** Single project-root `uv venv` with modern PyTorch (torch ≥ 2.4, CUDA 12.x). Both repos' READMEs request ancient PyTorch (1.6 / 1.9) inside conda envs — those won't even install cleanly on modern drivers, and we don't need their training pipelines. We use the model code only, and both subtrees only use standard `nn.Module` APIs (no exotic CUDA extensions). RAFT's optional `alt_cuda_corr` is not built (not needed unless we hit a VRAM ceiling). Per the brief: "Don't force `uv` if it breaks an existing setup" — but here both repos' setups would break on modern hardware regardless, so `uv` is the cleaner call.
9. **DataParallel checkpoint keys.** `raft-things.pth` keys are prefixed with `module.`. The RAFT wrapper strips that prefix on load (or wraps with `nn.DataParallel`). Single-GPU inference uses raw module — strip the prefix, no DataParallel.
10. **Sintel `occlusions/` polarity.** Convention: pixel value 255 = occluded, 0 = not occluded. Confirm against `flow_code/README.txt` at step 8; mis-polarity would flip Matched vs Unmatched EPE.
11. **`invalid/` semantics.** Pixel value 255 = invalid GT (unreliable), should be **excluded** from all EPE / Bad-X / matched-unmatched / boundary computations. The `epe.py` API will accept this mask once at construction and apply it everywhere — easy to forget if added later.
12. **Floating-point non-determinism.** Different runs on the same GPU can disagree by ~1e-5 due to cuDNN kernel selection. Not material at EPE granularity, but two re-runs won't bit-match. Mention this if the report claims reproducibility.
13. **Disk usage ~15 GB.** Predictions + cached masks fit comfortably on the local disk. Logged here so it doesn't surprise anyone.
14. **Methodology numbering mismatch.** Brief refers to "Section 4 item 13"; §4 has 9 items in the on-disk file. Trivial, just noting.
15. **Implementation NOT to add without approval.** AEPE (Angular Endpoint Error), Fl-all (KITTI-style), per-region IoU between predicted occlusion (fwd-bwd) and Sintel native — all reasonable, none in methodology. I will not add them unless approved.

---

## GO / NO-GO

**Recommendation: GO**, with two soft asks before coding starts:

- **Confirm headline-checkpoint choice** (committed: Things-trained for both). If you want Sintel-finetuned headlines instead, flip §5 item 5.
- **Pin down the Oei 2026 GMFlow 0.94 reference** (§5 item 4) so step 7's sanity check has the right target alongside the GMFlow paper's 1.495.

Neither is blocking. Once you give the word, implementation proceeds in the order in §3.
