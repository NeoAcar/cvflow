# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project-specific context (cvflow)

### What this repo is

Topic E final project — **inference-only** comparison of RAFT (Teed & Deng 2020) and GMFlow (Xu et al. 2022) on MPI-Sintel + Middlebury + (Phase 2) RobustSpring. **No training.** Methodology lives in `docs/eval_methodology.md`; the original feasibility plan lives in `docs/initial_plan.md`; Phase 1 numbers live in `docs/phase1_results.md`.

### Environment + paths

- Single `uv venv` at repo root: `.venv/`. Python 3.11. PyTorch 2.6.0+cu124.
- **Always activate the venv and set `PYTHONPATH=src` before running anything in `src/cvflow/`.**
  ```bash
  source .venv/bin/activate && export PYTHONPATH=src
  ```
- Do **not** use the bundled `environment.yml` in `gmflow/gmflow/` or the conda recipe in `RAFT/RAFT/README.md` — they pin PyTorch 1.6/1.9 which won't install on modern drivers.

### Datasets on disk

- Sintel: `datasets/Sintel/training/{clean,final,flow,occlusions,invalid,albedo,flow_viz}/` — 23 sequences, 1041 pairs per pass at 1024×436. `motion_boundaries/` is **not** present; the Disc mask is derived (Sobel on GT flow + 9×9 dilate).
- Middlebury: `datasets/Middleburry/{other-data,other-gt-flow}/` — 8 of 12 sequences have GT. Note the typo `Middleburry` (double-r) — keep it; that's the on-disk path.
- Middlebury GT uses the sentinel `|u| ≥ 1e9` / `|v| ≥ 1e9` for invalid pixels — **must mask before computing EE/AE** (`cvflow.metrics.middlebury.gt_valid_mask`).
- Sintel invalid mask: pixel value `255 = invalid GT`. Apply with `invalid == 0` as the "valid" predicate.
- Sintel occlusion mask: pixel value `255 = occluded`. Convention: matched = valid ∧ occlusion == 0; unmatched = valid ∧ occlusion == 255.

### Model wrappers

- `cvflow.models.raft_wrapper.RaftWrapper` and `cvflow.models.gmflow_wrapper.GMFlowWrapper` share a single `predict(img1_u8, img2_u8) -> float32[H,W,2]` contract.
- **Headline checkpoints: `raft-things.pth` and `gmflow_things-e9887eda.pth`** (Things-trained, zero-shot framing). Do not silently switch to Sintel-finetuned variants — that contaminates the Sintel→Middlebury generalization claim (hypothesis 10).
- RAFT headline iters = **32** (matches Teed & Deng inference setting). GMFlow uses `attn_splits=[2], corr_radius=[-1], prop_radius=[-1], padding_factor=16` (basic, no refinement).
- RAFT checkpoint keys are prefixed `module.` (saved through `nn.DataParallel`); wrapper strips the prefix on load.
- **`utils.utils` name collision** between RAFT/core and gmflow: both repos ship a `utils/` directory but only RAFT's has `__init__.py`. We inline the `InputPadder` in `src/cvflow/models/_padder.py` to avoid the collision. **Do not** re-add `from utils.utils import InputPadder` to either wrapper — it will break in import-order-dependent ways.

### Mask thresholds (Baker et al. 2011 conventions)

| Mask | Source | Threshold | Dilation |
|---|---|---|---|
| Disc | `‖∇u‖₁ + ‖∇v‖₁` on GT flow | `> 1.0` | 9×9 |
| Untex | Sobel gradient magnitude on I₁ | `< 5.0` | 3×3 |
| boundary F-score tolerance | dilation of true-positive band | — | `2*tol_px+1 = 5×5` |

Thresholds are CLI flags on `eval_from_saved.py`. Change them in one place; predictions don't need re-running.

### Conventions for new code

- Output flow is always `float32[H,W,2]`, frame1→frame2 displacement in pixels.
- Predictions are saved as `.npy` (raw `float32`) at `results/predictions/<model_tag>/<dataset>/<pass>/<seq>/frame_NNNN.npy`. Don't re-shape into HDF5 unless asked.
- Sintel sequence ordering: `sorted(...)` glob — alphabetical is the convention everywhere in the pipeline.
- Don't write CPU fallbacks for things that need GPU. If something OOMs on the 4 GB 3050 Ti, the call is "move to Colab A100", not "add a CPU branch".
- Add metrics only after surfacing them — do not silently introduce new metrics not in `docs/eval_methodology.md` §4.

### Scope

- **Phase 1 (complete):** Sintel clean + final, Middlebury, iter sweep, latency/VRAM, photometric residual + critique-driven revisions (GMFlow-refine three-way ablation, sequence-level bootstrap CIs, boundary-F1 threshold sweep, quantization-residual histograms, blur–motion confound check, alternative H10 normalization, H9 retraction). All §4 items 1–10 and 13 produced.
- **Out of scope:** RobustSpring corruption suite — methodology §1.6, §4 items 11–12, hypothesis 8. Descoped, not deferred. Do not assume future runs will produce these numbers.