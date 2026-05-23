# Optical Flow Robustness Evaluation Methodology
### RAFT vs GMFlow on MPI-Sintel, Middlebury, and RobustSpring

> Companion methodology document for the Final Project proposal
> (Topic E — Flow Estimation Robustness). This document expands on
> the proposal with concrete metrics, failure-region detectors,
> testable hypotheses, and the additional measurements that should
> appear in the methodology section.
>
> **Conventions cross-referenced with:**
> - Baker, Scharstein, Lewis, Roth, Black & Szeliski (2011),
>   *A Database and Evaluation Methodology for Optical Flow*,
>   IJCV 92(1), 1–31 — the Middlebury benchmark paper.
>   Henceforth "Baker et al. 2011".
> - Oei, Schmalfuss, Mehl, Bartsch, Agnihotri, Keuper, Bulling &
>   Bruhn (2026), *RobustSpring: Benchmarking Robustness to Image
>   Corruptions for Optical Flow, Scene Flow and Stereo*,
>   ICLR 2026. Henceforth "Oei et al. 2026".

---

## Table of Contents

1. [Quality Metrics](#1-quality-metrics)
2. [Failure Modes and How to Detect Them](#2-failure-modes-and-how-to-detect-them)
3. [Pairwise Hypotheses — RAFT vs GMFlow](#3-pairwise-hypotheses--raft-vs-gmflow)
4. [Must-Measure Additions Not in the Original Proposal](#4-must-measure-additions-not-in-the-original-proposal)

---

## 1. Quality Metrics

There are **two fundamentally different classes** of metrics. The
GT-based metrics in §1.1–1.5 quantify accuracy against ground truth.
The corruption-robustness metric in §1.6 (Oei et al. 2026) measures
prediction *stability* under input perturbation and requires no GT.
Oei et al. 2026 (Sect. 3.2) emphasize that accuracy and robustness
are *competing* axes — a model that always returns zero is perfectly
robust but useless — so both must be reported separately.

### 1.1 Per-pixel error measures (GT-based)

| Measure | Definition | Notes |
|---|---|---|
| **EPE** (End-Point Error) | `‖flow_pred − flow_gt‖₂ = √((u−uGT)² + (v−vGT)²)` | Preferred by Baker et al. 2011 (Sect. 5.2.1) |
| **AE** (Angular Error) | `cos⁻¹((1 + u·uGT + v·vGT) / (√(1+u²+v²) · √(1+uGT²+vGT²)))` | Historical standard from Barron et al. 1994. **Downweights errors in large flows** — Baker et al. 2011 show this can hide gross failures (Urban building example, AE rank 6 vs EE rank 20). Report alongside EE for backward-compatibility, but treat EE as the primary metric. |

### 1.2 Aggregate statistics (GT-based)

Both benchmarks use these — report on the same flow output to compare
with prior work on either side.

| Statistic | Middlebury convention (Baker et al. 2011) | Sintel/general convention |
|---|---|---|
| Mean and SD | Standard | Standard |
| **Robustness RX** (% pixels with error > X) | **R0.5, R1.0, R2.0** for EE in pixels; R2.5, R5.0, R10.0 for AE in degrees | Often called **Bad-1px, Bad-3px, Bad-5px** (Fl-all family). Same idea, different thresholds. |
| **Accuracy AX** (error at Xth percentile, sorted low→high) | **A50, A75, A95** for flow errors | Less common in newer papers but still informative |
| **Catastrophic failure** | Not standard, but compute % with EPE > 10 px | Same |

Report **both** Middlebury-style (R0.5/R1.0/R2.0) and Sintel-style
(Bad-1/3/5px) thresholds — costs nothing extra and lets you compare
with results in either literature.

### 1.3 Speed-bucket EPE (Sintel convention)

Sintel partitions pixels by GT flow magnitude:

```python
speed = np.linalg.norm(gt_flow, axis=-1)
mask_slow = speed < 10                          # s0-10
mask_mid  = (speed >= 10) & (speed < 40)        # s10-40
mask_fast = speed >= 40                         # s40+
```

Middlebury does not use these buckets directly — its sequences have
smaller motion ranges (mostly < 20 px). But on Sintel they are the
official reporting standard.

### 1.4 Region-specific reporting (Baker et al. 2011, Sect. 4.3)

Compute every metric above on three masks:

- **All** — every pixel where GT flow is reliably defined
- **Disc** — motion discontinuities (defined in Sect. 2.4 below)
- **Untext** — textureless regions (defined in Sect. 2.1 below)

This is the Middlebury reporting standard and is also useful on Sintel.

### 1.5 Auxiliary metrics

| Metric | Definition | What it tells us |
|---|---|---|
| **F-score @ flow edges** | F-score between predicted and GT flow edge maps | Quantifies sharpness independent of EPE magnitude |
| **Inference latency + VRAM** | Wall-clock time and peak GPU memory per frame pair | Practical comparison — missing from the proposal |

### 1.6 Corruption robustness metric (GT-free, Oei et al. 2026)

This is the **standard metric for the RobustSpring benchmark** and
must be used when reporting on it. It quantifies *stability* of a
model's prediction under corruption — no ground truth required.

For clean input `I` and corrupted input `I_c`, with prediction
function `f`, Oei et al. 2026 (Eq. 2) define:

$$R^c_M = M\big[f(I),\, f(I_c)\big]$$

where `M` is any distance metric. Concretely (Oei et al. 2026,
Sect. 3.2, with similar conventions to the Spring benchmark):

- **R_EPE^c**: average EPE between clean and corrupted predictions
  (Oei et al. 2026, Eq. 3):

  $$R^c_{EPE} = \frac{1}{|\Omega|}\sum_{i \in \Omega} \big\| f_i(I) - f_i(I_c) \big\|$$

- **R_1px^c**: percentage of pixels where clean vs corrupted
  prediction differs by more than 1 px
- **R_Fl^c**: KITTI Fl-style outlier rate between predictions

**Conceptual note (important):** Oei et al. 2026 (Sect. 3.2) deliberately
omit the Lipschitz denominator `‖I − I_c‖` because RobustSpring
equalizes its corruption strengths via SSIM (≥ 0.7 for most, ≥ 0.2
for noise — Sect. 3.1) so all corrupted images deviate by a similar
perceptual amount. Lower `R^c_M` = more stable = more robust.

**Ranking across the 20 corruptions** — Oei et al. 2026 (Sect. 3.2)
use three aggregation strategies:

- **Average** — sensitive to extreme outliers (rain dominates)
- **Median** — reduces impact of extreme corruption-specific failures
- **Schulze voting** (Schulze 2018) — pairwise preference aggregation,
  used previously in Robust Vision Challenges

Their finding: GMFlow ranks 2nd on Average but 5th on Median/Schulze
(Oei et al. 2026, Table 1) — meaning its strong average is driven by
specific corruption types, not uniform performance. **Report all
three** for our RAFT vs GMFlow comparison.

**Subsampling for efficiency** — Oei et al. 2026 (Sect. 3.2) evaluate
on only **0.05% of pixels** (their refined version of Spring's official
subsampling). This is essential: running full-resolution Spring/RobustSpring
on 20 corruptions can take many hours; subsampling brings it to ~1 hour.

---

## 2. Failure Modes and How to Detect Them

For each failure mode below, we list how to **identify the region**
on which to evaluate. Some regions come with a GT-provided mask;
others must be derived from image content or from GT flow itself.

### 2.1 Textureless regions (sky, walls, ground) — the **Untext** mask

Defined in Baker et al. 2011 Sect. 4.3 as: gradient of the image,
thresholded, then dilated with a **3×3 box**. The pixels excluded
from the All mask are also excluded here.

For our derivation from image content (no precomputed mask on Sintel):

- **Structure-tensor method** (preferred):
  1. Compute image gradients `Ix`, `Iy`.
  2. In a local window (e.g. 7×7) form
     `M = Σ [[Ix²,  IxIy], [IxIy,  Iy²]]`.
  3. Pixels with `λ_min(M) < τ` are textureless.
     Threshold `τ` is calibrated empirically by visualization.
- **Simpler proxy** (matches Baker et al. 2011's definition more closely):
  gradient magnitude `‖∇I‖ < τ`, then dilate with a 3×3 box.

**Important caveat from Baker et al. 2011 (Sect. 5.2.3):**
*"Textureless regions seem to be no problem for today's methods,
essentially all of which optimize a global energy."* This was their
finding on Middlebury circa 2009 with classical methods. The
interesting question for our study is whether RAFT (iterative
refinement) and GMFlow (single-pass global matching) behave
differently in textureless regions — both rely on global propagation
mechanisms but via very different machinery. Don't assume both will
fail catastrophically; the more nuanced question is which propagates
information into ambiguous regions more reliably.

### 2.2 Occlusions

- **Direct**: Sintel ships `occlusions/frame_xxxx.png` per pair — use it.
- **Cross-check (works without GT)**: forward–backward consistency.
  `flow_fwd(x) + flow_bwd(x + flow_fwd(x)) ≈ 0` for non-occluded pixels;
  large residual → occlusion candidate. Baker et al. 2011 use exactly
  this technique to define the GT masks in their fluorescent-texture
  data (Sect. 3.1).
- **Reporting**: Matched EPE vs Unmatched EPE.

### 2.3 Large displacement / fast motion

- **From GT directly**: `|gt_flow| > 40 px` → Sintel `s40+` bucket.
- **By sequence**: `market_2`, `ambush_5`, `temple_3` are inherently
  fast — report per-sequence numbers, not only averages.

Note: Middlebury's maximum motions are much smaller (up to ~10 px
on hidden-texture data, up to 35 px on `Urban`). The fast-motion
regime is primarily a Sintel/Spring story.

### 2.4 Motion boundaries — the **Disc** mask

Defined in Baker et al. 2011 Sect. 4.3 as: gradient of the
ground-truth flow field, threshold the magnitude, then dilate with a
**9×9 box**. If GT flow is unavailable, frame differencing is used
as a fallback. The pixels excluded from the All mask are also
excluded here.

For derivation:

- **Direct**: Sintel provides `motion_boundaries/` masks pre-computed.
- **Derived (Baker-style)**: apply Sobel/Canny to the GT flow
  magnitude, threshold `‖∇u‖ + ‖∇v‖ > τ`, then dilate with a 9×9 box
  using `cv2.dilate`.
- **Reporting**: EPE on this mask plus boundary F-score.

Baker et al. 2011 (Sect. 5.2.3) report that *Disc* regions remain the
hardest task for optical flow algorithms — errors are uniformly much
higher than in *All*, while *Untext* errors are typically lowest.
This is the single most important regime for distinguishing RAFT from
GMFlow.

### 2.5 Illumination change, shadows, specularities

No native mask, but detectable from photometric residual.

- Warp frame 1 to frame 2 using GT flow and take pixelwise difference:
  `r(x) = |I₁(x) − I₂(x + flow_gt(x))|`.
- Non-occluded pixels with `r(x) > τ` violate brightness constancy.
  Errors here are "the dataset's fault, not the model's" — useful for
  separating model failure from inherently hard inputs.
- **Sintel Clean vs Final** at the same pixel also localizes
  illumination/blur effects (Final adds them, Clean does not).

### 2.6 Motion blur

- **From the Final pass**: low local Laplacian variance,
  `var(Laplacian(I, 3×3)) < τ`, indicates blur.
- **From speed**: high `|gt_flow|` pixels in the Final pass are
  likely blurred (the renderer blurs fast-moving objects).
- **Cleanest path**: **RobustSpring**, where blur is an explicit
  corruption with severity levels — isolates the effect properly
  (Oei et al. 2026, Sect. 3.1, lists 5 blur types: defocus, gaussian,
  glass, motion, zoom).

### 2.7 Noise, weather, sensor corruptions

- Use **RobustSpring** directly: 20 corruption types organized into
  5 families — blur, color, noise, quality, weather — applied in a
  time-, stereo-, and depth-consistent manner (Oei et al. 2026,
  Sect. 3.1 and Fig. 1). Each corruption is calibrated via SSIM.
- Sintel Final mixes these together — only an aggregate
  "Final − Clean" delta is available there, not a per-corruption split.

### 2.8 Defocus and atmospheric effects (Sintel Final)

- Low Laplacian variance combined with small GT flow magnitude is a
  defocus candidate.
- Atmospheric effects are qualitative only on Sintel; on RobustSpring,
  fog/frost/rain/snow/spatter are explicit corruption labels.

---

## 3. Pairwise Hypotheses — RAFT vs GMFlow

Each hypothesis is stated as "model X is better at condition Y",
paired with a concrete measurement.

| # | Hypothesis | How to measure |
|---|---|---|
| 1 | **GMFlow is better at large displacements** — global matching finds distant correspondences in a single pass. | EPE on `s40+`; additional bin for `|gt_flow| > 60`. Note: AE will downweight these errors per Baker et al. 2011 (Sect. 5.2.1); use EE. |
| 2 | **RAFT is better at sub-pixel accuracy** — iterative refinement converges to fine accuracy. | R0.5 / Bad-1px on matched + textured pixels; also median EPE and A50. |
| 3 | **RAFT produces sharper motion boundaries** — local correlation volume preserves edges. | EPE on the Disc mask (9×9 dilation) + boundary F-score. Disc is the regime where models differ most (Baker et al. 2011, Sect. 5.2.3). |
| 4 | **GMFlow handles occlusions better** — attention propagates from distant unoccluded neighbors. | EPE on the Unmatched mask (Sintel `occlusions/`). |
| 5 | **Both handle textureless regions adequately, but via different mechanisms — measure how they propagate** — Baker et al. 2011 (Sect. 5.2.3) report Untext is *not* the bottleneck for global-energy methods. The question is whether RAFT's iterative refinement and GMFlow's attention behave differently. | EPE on Untext mask (gradient threshold + 3×3 dilation); ratio to each model's All EPE. Expected ratio < 1.5 for both; the *relative* ratio is what's interesting. |
| 6 | **RAFT iteration count trades accuracy for latency; GMFlow is single-pass** — characterise the trade-off. | Sweep RAFT iters ∈ {4, 8, 12, 32}; plot EPE vs latency. GMFlow is a single point on the same plot. |
| 7 | **Clean → Final degradation is smaller for GMFlow** — feature matching is more tolerant to brightness change. | `ΔEPE = EPE_Final − EPE_Clean` per model, with per-sequence distribution. |
| 8 | **RAFT's catastrophic weakness is weather; GMFlow's relative weakness is noise** — Oei et al. 2026 (Table 1, Sect. 4.1) report RAFT R^c_EPE = 42.41 on rain vs GMFlow R^c_EPE = 8.60; on snow, RAFT 7.16 vs GMFlow 3.60. For noise, both degrade but GMFlow remains comparable to or slightly better than vanilla RAFT (Gaussian: 4.70 vs 7.43; Impulse: 6.64 vs 6.51). Oei et al. 2026 attribute this to GMFlow's global matching being noise-sensitive within the transformer family, while stacked refinement models (e.g. SEA-RAFT) uniquely resist noise. The hypothesis to *test* (not assume) is whether these benchmark trends replicate in our pipeline. | Per-corruption R^c_EPE on RobustSpring using Eq. 2 of Oei et al. 2026 (clean prediction vs corrupted prediction); corruption × model heatmap; report Average, Median, and Schulze rankings (Oei et al. 2026, Sect. 3.2). |
| 9 | **GMFlow may blow up in VRAM at higher resolutions** — attention is O(N²). | Peak VRAM and runtime as a function of input resolution (Middlebury / Spring). |
| 10 | **Cross-dataset generalization (Sintel → Middlebury) drops for both, but unequally.** | Middlebury EPE using the Sintel-trained checkpoint, normalized by each model's Sintel-domain EPE. Baker et al. 2011 (Sect. 5.2.4) found dataset correlations are low — strong on Sintel ≠ strong on Middlebury. |

---

## 4. Must-Measure Additions Not in the Original Proposal

These items belong in the methodology section. Graders will look for
many of them; without them the study reads as "we ran two models and
averaged EPE".

1. **Speed-bucket analysis (`s0-10`, `s10-40`, `s40+`)** — Sintel's
   official reporting convention. Global EPE alone is insufficient.
2. **Matched / Unmatched split** — occlusion behavior dominates the
   raw average and must be separated.
3. **Both Middlebury (R0.5/R1.0/R2.0) and Sintel-style (Bad-1/3/5px)
   robustness thresholds** — costs nothing, enables comparison with
   either body of prior work. Also report A50/A75/A95 percentile
   accuracies per Baker et al. 2011.
4. **Angular Error reported alongside EPE** — historical metric, but
   note Baker et al. 2011's finding that AE downweights errors in
   large-motion regions (Urban building example). Use EE as the
   primary headline.
5. **Boundary F-score** — turns the qualitative "RAFT looks sharper"
   observation into a number.
6. **Inference latency and VRAM** — measured at 1024×436 for both
   models. Practical and trivial to add.
7. **RAFT iteration sweep** — visualizes RAFT's compute-vs-accuracy
   curve and forces an explicit, fair comparison setting against
   single-pass GMFlow. State clearly whether 12 or 32 iters is used
   as the headline number.
8. **Photometric-residual analysis** — separates "the model failed"
   from "the GT itself is hard" (brightness-constancy violations).
9. **Per-sequence reporting** instead of just dataset averages. Use
   characteristic clips: `market_2`, `ambush_5` (fast),
   `cave_4` (textureless), `bandage_2` (occlusion). Baker et al. 2011
   (Sect. 5.2.4) showed cross-dataset correlations are low —
   per-sequence breakdown is essential.
10. **Sintel Clean vs Final paired delta** — report the `ΔEPE`
    distribution per sequence and per pixel, not only the global mean.
11. **RobustSpring corruption robustness `R^c_M`** — the GT-free
    stability metric from Oei et al. 2026 (Eq. 2). Report all three
    flavors (`R^c_EPE`, `R^c_1px`, `R^c_Fl`) for parity with their
    Table 1, and all three rankings (Average, Median, Schulze)
    per Oei et al. 2026 (Sect. 3.2).
12. **RobustSpring subsampling** — use the official 0.05% subsample
    (Oei et al. 2026, Sect. 3.2) rather than full resolution; brings
    runtime to ~1 hour. State which subsampling was used.
13. **Sanity-check against published Things-trained Sintel numbers**
    — before running the full evaluation, validate the pipeline by
    reproducing the zero-shot Sintel-train numbers from the original
    papers. Both checkpoints are Chairs+Things only (no Sintel
    finetuning), matching our zero-shot framing:

    | Model | Checkpoint | Sintel Clean (train) EPE | Sintel Final (train) EPE |
    |---|---|---|---|
    | RAFT | `raft-things.pth` | **1.43** | **2.71** |
    | GMFlow (basic, no refinement) | `gmflow_things-e9887eda.pth` | **~1.50** | **~2.96** |

    RAFT numbers from Teed & Deng (2020), Table 1 (C+T row);
    confirmed in the official repo (princeton-vl/RAFT issue #124).
    GMFlow numbers from Xu et al. (2022), Table 3; the official
    `gmflow/scripts/evaluate.sh` reports 1.495 on Sintel clean train
    for the basic checkpoint.

    If your pipeline diverges from these by more than ~10%, something
    is wrong (input preprocessing, padding, normalization, checkpoint
    mismatch). Note: the `ε_clean` column in Oei et al. 2026, Table 1
    is on **Spring**, not Sintel — do not use those numbers (RAFT
    1.48, GMFlow 0.94) for Sintel sanity-checking.

---

### Suggested reporting hierarchy (one-paragraph version)

If forced to compress the above into a single methodology paragraph,
the priority order is:

> **(1)** speed-bucket EPE, **(2)** matched/unmatched split,
> **(3)** R0.5/R1.0/R2.0 + Bad-1/3/5px, **(4)** Disc-mask EPE +
> boundary F-score, **(5)** RobustSpring per-corruption `R^c_M`
> with Average/Median/Schulze rankings, **(6)** latency and VRAM,
> **(7)** RAFT iteration sweep, **(8)** per-sequence breakdown.

Without these, the study reduces to averaging EPE over two models on
one dataset.

---

### References

Baker, S., Scharstein, D., Lewis, J.P., Roth, S., Black, M.J., &
Szeliski, R. (2011). A Database and Evaluation Methodology for
Optical Flow. *International Journal of Computer Vision*, 92(1),
1–31. [http://vision.middlebury.edu/flow/](http://vision.middlebury.edu/flow/)

Oei, V., Schmalfuss, J., Mehl, L., Bartsch, M., Agnihotri, S.,
Keuper, M., Bulling, A., & Bruhn, A. (2026). RobustSpring:
Benchmarking Robustness to Image Corruptions for Optical Flow, Scene
Flow and Stereo. In *International Conference on Learning
Representations (ICLR 2026)*.