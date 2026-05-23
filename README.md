# cvflow

Inference-only evaluation of RAFT and GMFlow on MPI-Sintel and Middlebury for the Topic E final project (Flow Estimation Robustness). Phase 2 adds RobustSpring.

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

Phase 1 uses `raft-things.pth` and `gmflow_things-e9887eda.pth` — the Chairs+Things zero-shot checkpoints. The other variants (`*-sintel`, `*-kitti`, `*-chairs`, `*-with-refine-*`) are downloaded but unused at this stage.

## Datasets

The Sintel training tarball goes under `datasets/Sintel/training/{clean,final,flow,occlusions,invalid,...}/`. The Middlebury "other" set goes under `datasets/Middleburry/{other-data,other-gt-flow}/`. **`motion_boundaries/` is missing from the on-disk Sintel** — we derive the Disc mask from GT flow (see methodology §2.4).

## Run

```bash
source .venv/bin/activate
export PYTHONPATH=src

python -m cvflow.runners.run_sintel_eval --model raft   --pass clean   # ~14 min on RTX 3050 Ti
python -m cvflow.runners.run_sintel_eval --model gmflow --pass clean   # ~6 min
python -m cvflow.runners.run_sintel_eval --model raft   --pass final
python -m cvflow.runners.run_sintel_eval --model gmflow --pass final

# Offline (CPU) — adds Disc/Untex/F-score from saved .npy predictions:
python -m cvflow.runners.eval_from_saved --pred-root results/predictions/raft-raft-things-iter32   --pass clean
python -m cvflow.runners.eval_from_saved --pred-root results/predictions/gmflow-gmflow_things-e9887eda --pass clean

python -m cvflow.runners.run_middlebury --model both
python -m cvflow.runners.run_raft_itersweep
python -m cvflow.runners.run_latency_vram --n 50
python -m cvflow.runners.run_photometric
```

Predictions are written to `results/predictions/<model_tag>/<dataset>/<pass>/<seq>/frame_NNNN.npy` as raw `float32 [H,W,2]`. ~3.5 GB per `(model, pass)`.

## Reading order

1. `docs/eval_methodology.md` — what we measure and why
2. `docs/initial_plan.md` — feasibility check + repo design + risks (planning gate)
3. `docs/phase1_results.md` — Phase 1 numbers and per-hypothesis verdict
4. `progress.md` — what's done, what's pending
5. `technical.md` — repo structure + per-file purpose
