# cvflow

Inference-only evaluation of RAFT and GMFlow on MPI-Sintel and Middlebury for the Topic E final project (Flow Estimation Robustness). RobustSpring / corruption-robustness work is **out of scope** for this study — H8 in `docs/phase1_results.md` §8 is marked "not pursued".

## Layout

```
src/cvflow/         pipeline code (datasets, models, masks, metrics, runners)
RAFT/RAFT/          upstream RAFT repo (princeton-vl/RAFT) — model code + checkpoints
gmflow/gmflow/      upstream GMFlow repo (haofeixu/gmflow) — model code + checkpoints
datasets/           MPI-Sintel + Middlebury 'other' set on disk
docs/               methodology, plan, Phase 1 results
results/            saved predictions (.npy) and metric tables
cache/              derived mask cache (currently unused — masks recomputed)
logs/               per-run logs
```

## Environment

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
uv pip install numpy opencv-python pillow scipy einops matplotlib gdown
```

Both repos' bundled `environment.yml`/conda recipes target PyTorch 1.6/1.9 — too old to install cleanly on modern drivers. We use the model code only, via `PYTHONPATH=src` + `sys.path` injection in the wrappers.

## Checkpoints

```bash
# RAFT
cd RAFT/RAFT && ./download_models.sh            # Dropbox zip, ~85 MB

# GMFlow
gdown '1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2' -O /tmp/gmflow_pretrained.zip
unzip -d gmflow/gmflow /tmp/gmflow_pretrained.zip
```

Phase 1 uses three checkpoints, all Chairs+Things zero-shot:

- `raft-things.pth` (RAFT, 32 inference iterations)
- `gmflow_things-e9887eda.pth` (GMFlow basic — no refinement)
- `gmflow_with_refine_things-36579974.pth` (GMFlow with-refinement — added in the critique-driven revision round for a capacity-matched comparison; see `docs/phase1_results.md` §11)

Other variants (`*-sintel`, `*-kitti`, `*-chairs`) are downloaded but unused.

## Datasets

The Sintel training tarball goes under `datasets/Sintel/training/{clean,final,flow,occlusions,invalid,...}/`. The Middlebury "other" set goes under `datasets/Middleburry/{other-data,other-gt-flow}/`. **`motion_boundaries/` is missing from the on-disk Sintel** — we derive the Disc mask from GT flow (see methodology §2.4).

## Run

```bash
source .venv/bin/activate
export PYTHONPATH=src

python -m cvflow.runners.run_sintel_eval --model raft   --pass clean   # ~14 min on RTX 3050 Ti
python -m cvflow.runners.run_sintel_eval --model gmflow --pass clean   # ~6 min
python -m cvflow.runners.run_sintel_eval --model gmflow --gmflow-refine --pass clean  # ~16 min
python -m cvflow.runners.run_sintel_eval --model raft   --pass final
python -m cvflow.runners.run_sintel_eval --model gmflow --pass final
python -m cvflow.runners.run_sintel_eval --model gmflow --gmflow-refine --pass final

# Offline (CPU) — full mask suite + per-seq JSON dump for bootstrap:
python -m cvflow.runners.eval_from_saved --pred-root results/predictions/raft-raft-things-iter32 \
  --pass clean --dump-json results/per_seq_stats/raft_clean.json

python -m cvflow.runners.run_middlebury --model both
python -m cvflow.runners.run_middlebury --model gmflow --gmflow-refine
python -m cvflow.runners.run_raft_itersweep
python -m cvflow.runners.run_latency_vram --n 50
python -m cvflow.runners.run_photometric
python -m cvflow.runners.delta_epe_maps --pred-root results/predictions/raft-raft-things-iter32
python -m cvflow.runners.run_vram_resolution               # H9 — needs ≥16 GB GPU for clean reading
python -m cvflow.runners.run_fwdbwd_occlusion --model raft \
  --fwd-cache results/predictions/raft-raft-things-iter32/sintel/clean
python -m cvflow.runners.speed_curves
python -m cvflow.runners.middlebury_correlation

# Critique-driven analyses (no GPU; require per-seq JSONs):
python -m cvflow.runners.bootstrap_compare \
  --a results/per_seq_stats/raft_clean.json \
  --b results/per_seq_stats/gmflow_refine_clean.json
python -m cvflow.runners.boundary_threshold_sweep
python -m cvflow.runners.quantization_check
python -m cvflow.runners.blur_motion_confound
```

Predictions are written to `results/predictions/<model_tag>/<dataset>/<pass>/<seq>/frame_NNNN.npy` as raw `float32 [H,W,2]`. ~3.5 GB per `(model, pass)`.

## Reading order

1. `docs/eval_methodology.md` — what we measure and why
2. `docs/initial_plan.md` — feasibility check + repo design + risks (planning gate)
3. `docs/phase1_results.md` — Phase 1 numbers and per-hypothesis verdict
4. `progress.md` — what's done, what's pending
5. `technical.md` — repo structure + per-file purpose
