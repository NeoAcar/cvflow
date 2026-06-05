# Derived masks — what we compute when Sintel doesn't ship the mask

Sintel ships GT flow, `invalid`, and `occlusions`. The four masks below — **Disc, Untex, Blur, Boundary-F1, Fwd-Bwd Occlusion** — are *not* in the dataset. We derive them from images or flow. This doc records exactly what each one does, kernel by kernel, threshold by threshold. Source files are linked.

Default thresholds match `docs/eval_methodology.md`. They are CLI knobs on `eval_from_saved.py` (`--disc-thresh`, `--untex-thresh`, `--blur-window`, `--blur-thresh`).

---

## 1. Disc — motion-discontinuity mask

**Source:** `src/cvflow/masks/motion_boundary.py::disc_mask`
**Input:** GT flow `float32[H, W, 2]`
**Default:** `grad_thresh = 1.0`, dilate kernel `9×9`

```python
u, v = gt_flow[..., 0], gt_flow[..., 1]
ux = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)   # ∂u/∂x  Sobel 3×3
uy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)   # ∂u/∂y
vx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
vy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
grad = |ux| + |uy| + |vx| + |vy|               # L1 sum (not magnitude)
high = (grad > 1.0)                            # px/px units
disc = cv2.dilate(high, np.ones((9, 9)))       # 9×9 box dilation
```

- **Kernel**: OpenCV Sobel 3×3 (i.e. `[[-1,0,1],[-2,0,2],[-1,0,1]]` for ∂/∂x).
- **Score**: L1 sum of all four partial derivatives (`|∇u|₁ + |∇v|₁`), not the L2 magnitude.
- **Threshold**: `1.0` px/px — pixels where total flow gradient exceeds 1 px change per pixel are "boundary-adjacent" before dilation.
- **Dilation**: 9×9 box (Baker et al. 2011 §4.3 convention).

---

## 2. Untex — textureless mask

**Source:** `src/cvflow/masks/textureless.py::untext_mask`
**Input:** I₁ as `uint8[H, W, 3]` RGB (or grayscale)
**Default:** `grad_thresh = 5.0`, dilate kernel `3×3`

```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # uint8
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad_mag = sqrt(gx² + gy²)                     # L2 magnitude (not Harris)
low = (grad_mag < 5.0)                         # intensity-gradient units (0..255 scale)
untex = cv2.dilate(low, np.ones((3, 3)))       # 3×3 box dilation
```

- **Kernel**: Sobel 3×3 on luminance.
- **Score**: L2 gradient magnitude (here we use magnitude, not L1 — different from Disc).
- **Threshold**: `5.0` on intensity gradient (input scale 0..255). Pixels with image gradient under 5 are textureless.
- **Dilation**: 3×3 box. *Note in the source*: Baker §4.3 dilates the *excluded* (high-gradient) pixels; we dilate the low-gradient region instead so the buffer is symmetric with Disc — see comment in `textureless.py:23-25`.
- **Not Harris**, not eigenvalue-based — pure Sobel magnitude.

---

## 3. Boundary F1 — predicted vs GT motion boundaries

**Source:** `src/cvflow/metrics/boundary_fscore.py::boundary_fscore`
**Input:** predicted flow + GT flow, both `float32[H, W, 2]`
**Default:** `grad_thresh = 1.0`, tolerance `tol_px = 2` (so dilation kernel `2·2+1 = 5×5`)

```python
# Same Sobel-L1 score as Disc, applied to BOTH predicted and GT flow:
p_raw = |Sobel(pred_u,x)| + |Sobel(pred_u,y)| + |Sobel(pred_v,x)| + |Sobel(pred_v,y)|
g_raw = |Sobel(gt_u,x)|   + |Sobel(gt_u,y)|   + |Sobel(gt_v,x)|   + |Sobel(gt_v,y)|
pred_b = (p_raw > 1.0)                          # predicted boundary mask
gt_b   = (g_raw > 1.0)                          # GT-derived boundary mask

k = 2 * tol_px + 1                              # = 5 for tol_px=2
pred_dil = cv2.dilate(pred_b, np.ones((k, k)))  # tolerance band on pred
gt_dil   = cv2.dilate(gt_b,   np.ones((k, k)))  # tolerance band on GT

tp_pred = ((pred_b == 1) & (gt_dil == 1)).sum()  # pred pixels near a GT boundary
tp_gt   = ((gt_b == 1)   & (pred_dil == 1)).sum()  # GT pixels near a pred boundary

precision = tp_pred / pred_b.sum()
recall    = tp_gt   / gt_b.sum()
F1 = 2 * P * R / (P + R)
```

- **Two derived masks**: predicted-boundary and GT-boundary are both from Sobel-L1 on flow, same `>1.0` threshold as Disc.
- **Tolerance band**: a predicted boundary pixel is TP if it falls within a 5×5 (tol=2) box around any GT boundary pixel. Standard BSDS / DAVIS practice — avoids penalizing 1-px localization offsets.
- **Precision/recall use asymmetric TP counts**: `tp_pred` for precision (predicted ∩ dilated-GT), `tp_gt` for recall (GT ∩ dilated-pred). Both bands are equal-size, so the asymmetry is only in what is counted, not in tolerance width.
- **No classifier output is used** — the model never says "this pixel is a boundary." We *derive* a boundary mask from its predicted flow and compare to a derivation from GT.
- §11b sensitivity sweep varies `grad_thresh ∈ {0.5, 1.0, 2.0}`, tolerance stays at 2 px.

---

## 4. Blur / defocus mask

**Source:** `src/cvflow/masks/blur.py::blur_mask`
**Input:** I₁ as `uint8[H, W, 3]` (or grayscale)
**Default:** `window = 7`, `var_thresh = 20.0`

```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
lap  = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)      # 3×3 Laplacian
mean    = cv2.boxFilter(lap,     ksize=(7, 7), normalize=True)
mean_sq = cv2.boxFilter(lap*lap, ksize=(7, 7), normalize=True)
local_var = max(mean_sq − mean², 0)                  # local var of Laplacian
blur = local_var < 20.0
```

- **Kernel**: 3×3 Laplacian (`[[0,1,0],[1,-4,1],[0,1,0]]`), then a 7×7 box filter twice to get local variance per pixel.
- **Per-pixel local variance**: classic Pech-Pacheco "variance of Laplacian" sharpness heuristic, but applied *per-pixel* over a sliding 7×7 window — not a single global frame score.
- **Threshold**: `var < 20` ⇒ "blur-like" (low local Laplacian energy ≈ defocus / motion blur). Order-of-magnitude pick, not calibrated against GT defocus.
- **Caveats** (already noted in §11c of `phase1_results.md`):
  - No GT defocus exists; this is an unsupervised heuristic.
  - Correlates with motion magnitude across the 23 sequences at Pearson 0.644 / Spearman 0.520 — moderate confound, not pure. `mountain_1` (low motion + high blur fraction) confirms it isn't *only* a fast-motion proxy.
  - Per-sequence blur fractions can be reproduced from `results/per_seq_stats/*.json` via `sum_n[blur] / sum_n[all]` (the recent §7d fix used this).

---

## 5. Forward-backward consistency occlusion mask

**Source:** `src/cvflow/runners/run_fwdbwd_occlusion.py` (helpers: `warp_flow`, `fwdbwd_occlusion`)
**Input:** forward flow f₁₂ = model(I₁, I₂), backward flow f₂₁ = model(I₂, I₁) — both `float32[H, W, 2]`
**Default:** `alpha = 0.01`, `beta = 0.5` (Sundaram et al. 2010 / Meister et al. 2018 conventions)

```python
# Sample backward flow at warped forward locations: f21_at(x) = f21( x + f12(x) )
yy, xx = meshgrid(arange(H), arange(W), indexing="ij")
map_x = xx + f12[..., 0]
map_y = yy + f12[..., 1]
f21_at_x = cv2.remap(f21[..., 0], map_x, map_y, INTER_LINEAR, BORDER_REPLICATE)
f21_at_y = cv2.remap(f21[..., 1], map_x, map_y, INTER_LINEAR, BORDER_REPLICATE)
f21_at = stack([f21_at_x, f21_at_y], axis=-1)

# Inconsistency test:
diff_sq = ((f12 + f21_at) ** 2).sum(axis=-1)            # ‖f12 + f21_at‖²
norm_sq = (f12 ** 2).sum(axis=-1) + (f21_at ** 2).sum(axis=-1)
derived_occluded = diff_sq > (alpha * norm_sq + beta)   # 0.01·‖·‖² + 0.5
```

- **No new kernel** — just bilinear `cv2.remap` for warping (with `BORDER_REPLICATE`) and two squared L2 norms.
- **Test interpretation**: in a consistent (non-occluded) pixel, `f₁₂(x) + f₂₁(x + f₁₂(x)) ≈ 0`. The threshold `α·(‖f₁₂‖² + ‖f₂₁_at‖²) + β` scales with flow magnitude (so faster motion gets more slack) plus a constant absolute floor `β = 0.5 px²` for very slow regions.
- **Two model passes per Sintel pair**: forward predictions are loaded from cache (`--fwd-cache results/predictions/<tag>/sintel/clean`); backward is `model.predict(img2, img1)` recomputed live.
- **Evaluation** is against Sintel's native `occlusion == 255` mask. Precision/recall/F1/IoU are computed only on pixels where `invalid == 0` (so the GT comparison itself is GT-valid-only).
- §7c reported precision 0.654/0.752, recall 0.564/0.504, F1 0.606/0.603, IoU 0.435/0.432 for RAFT / GMFlow-basic.

---

## What's in the dataset, what we derive (recap)

| Mask | Source | Sintel native? | Where computed |
|---|---|---|---|
| `valid` | `invalid == 0` | ✓ shipped | `Sintel.pairs(...)` reads `invalid/<seq>/frame_*.png` |
| `matched` / `unmatched` | `occlusion == 0` / `== 255` | ✓ shipped | reads `occlusions/<seq>/frame_*.png` |
| Speed buckets `s0_1`, `s0_10`, `s10_40`, `s40+`, `s60+` | `‖gt_flow‖` per pixel | derived from GT | `SintelMetrics.update()` |
| **Disc** | Sobel-L1 of GT flow + 9×9 dilate | **derived** | `masks/motion_boundary.py` |
| **Untex** | Sobel-L2 of I₁ + 3×3 dilate | **derived** | `masks/textureless.py` |
| **Blur** | local var of 3×3 Laplacian in 7×7 window | **derived** | `masks/blur.py` |
| **Boundary F1** | Sobel-L1 on pred and GT flow, 5×5 tolerance band | **derived** | `metrics/boundary_fscore.py` |
| **Fwd-bwd occlusion** | `‖f₁₂ + f₂₁(x+f₁₂)‖² > 0.01·‖·‖² + 0.5` | **derived** (vs native ground truth) | `runners/run_fwdbwd_occlusion.py` |

All derived masks use the *same* GT flow + image inputs for every model, so cross-model comparisons within this study are internally consistent. Comparisons to external papers should note that our Disc / Boundary-F1 / Untex / Blur are derived, not native, so absolute numbers are not directly comparable to papers using shipped Sintel-extras masks or BSDS-style human-labeled boundaries.
