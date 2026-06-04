"""Generate report-ready optical-flow panels and region-score figures.

The figures are intentionally small in count and high in explanatory value:
input frame, GT flow, model flow, EPE maps, and the failure-region masks used
by the metrics. This is for slides/report figures, not full benchmark runs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cvflow.datasets.sintel import Sintel
from cvflow.masks.blur import blur_mask
from cvflow.masks.motion_boundary import disc_mask
from cvflow.masks.textureless import untext_mask


DEFAULT_SAMPLES = ("market_2:20", "ambush_5:20", "cave_4:20", "bandage_2:20")
REGION_ORDER = ("all", "matched", "unmatched", "s0_1", "s0_10", "s10_40", "s40+", "s60+", "disc", "untex", "blur")
GMFLOW_REFINE_PRESET = dict(
    padding_factor=32,
    attn_splits_list=[2, 8],
    corr_radius_list=[-1, 4],
    prop_radius_list=[-1, 1],
    num_scales=2,
    upsample_factor=4,
)


def flow_to_rgb(flow: np.ndarray, clip: float | None = None) -> np.ndarray:
    """Map flow to a Middlebury-style HSV color visualization."""
    u = flow[..., 0]
    v = flow[..., 1]
    mag, ang = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32), angleInDegrees=True)
    if clip is None:
        clip = float(np.percentile(mag, 99))
    clip = max(clip, 1e-6)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # OpenCV hue: 0..179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / clip * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def normalize_heat(values: np.ndarray, vmax: float | None = None) -> tuple[np.ndarray, float]:
    if vmax is None:
        finite = values[np.isfinite(values)]
        vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    return np.clip(values / vmax, 0, 1), vmax


def heat_to_rgb(norm: np.ndarray) -> np.ndarray:
    arr = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def relative_error_to_rgb(diff: np.ndarray, vmax: float | None = None) -> tuple[np.ndarray, float]:
    """Blue = RAFT has larger EPE, red = GMFlow has larger EPE."""
    finite = diff[np.isfinite(diff)]
    if vmax is None:
        vmax = float(np.percentile(np.abs(finite), 95)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    x = np.clip(diff / vmax, -1.0, 1.0)

    rgb = np.empty((*diff.shape, 3), dtype=np.float32)
    # Negative values: RAFT worse -> blue. Positive values: GMFlow worse -> red.
    neg = x < 0
    pos = x > 0
    rgb[...] = 245.0
    rgb[neg, 0] = 245.0 * (1.0 + x[neg])
    rgb[neg, 1] = 245.0 * (1.0 + x[neg])
    rgb[neg, 2] = 245.0
    rgb[pos, 0] = 245.0
    rgb[pos, 1] = 245.0 * (1.0 - x[pos])
    rgb[pos, 2] = 245.0 * (1.0 - x[pos])
    return np.clip(rgb, 0, 255).astype(np.uint8), vmax


def mask_overlay(img: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    out = img.astype(np.float32).copy()
    colors = {
        "unmatched": np.array([255, 70, 70], dtype=np.float32),
        "disc": np.array([255, 210, 40], dtype=np.float32),
        "untex": np.array([70, 170, 255], dtype=np.float32),
        "s40+": np.array([170, 80, 255], dtype=np.float32),
        "blur": np.array([40, 220, 120], dtype=np.float32),
    }
    for name, color in colors.items():
        m = masks[name]
        out[m] = 0.55 * out[m] + 0.45 * color
    return np.clip(out, 0, 255).astype(np.uint8)


def font(size: int = 18):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def tile(title: str, img: np.ndarray, width: int = 360) -> Image.Image:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    new_h = max(1, round(pil.height * width / pil.width))
    pil = pil.resize((width, new_h), Image.Resampling.BILINEAR)
    header = 34
    out = Image.new("RGB", (width, new_h + header), "white")
    out.paste(pil, (0, header))
    d = ImageDraw.Draw(out)
    d.text((8, 7), title, fill=(20, 20, 20), font=font(17))
    return out


def text_tile(lines: list[str], width: int = 360, height: int = 187) -> Image.Image:
    out = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(out)
    f = font(12)
    y = 6
    for line in lines:
        d.text((10, y), line, fill=(20, 20, 20), font=f)
        y += 14
    return out


def grid_image(tiles: list[Image.Image], cols: int = 4, pad: int = 14) -> Image.Image:
    rows = int(np.ceil(len(tiles) / cols))
    w = max(t.width for t in tiles)
    row_heights = [max(t.height for t in tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    out = Image.new("RGB", (cols * w + (cols + 1) * pad, sum(row_heights) + (rows + 1) * pad), (245, 245, 245))
    y_offsets = [pad]
    for h in row_heights[:-1]:
        y_offsets.append(y_offsets[-1] + h + pad)
    for i, t in enumerate(tiles):
        x = pad + (i % cols) * (w + pad)
        y = y_offsets[i // cols]
        out.paste(t, (x, y))
    return out


def parse_samples(samples: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for s in samples:
        if ":" not in s:
            raise ValueError(f"sample must be seq:idx, got {s!r}")
        seq, idx = s.split(":", 1)
        out.append((seq, int(idx)))
    return out


def get_pair(sintel_root: str, pass_: str, seq: str, idx: int):
    ds = Sintel(sintel_root, pass_=pass_)
    for pair in ds.pairs([seq]):
        if pair.idx == idx:
            return pair
    raise ValueError(f"could not find {pass_} pair {seq}:{idx}")


def masks_for_pair(pair) -> dict[str, np.ndarray]:
    valid = pair.invalid == 0
    speed = np.linalg.norm(pair.gt_flow, axis=-1)
    disc = disc_mask(pair.gt_flow)
    untex = untext_mask(pair.img1)
    blur = blur_mask(pair.img1)
    return {
        "all": valid,
        "matched": valid & (pair.occlusion == 0),
        "unmatched": valid & (pair.occlusion == 255),
        "s0_1": valid & (speed < 1),
        "s0_10": valid & (speed < 10),
        "s10_40": valid & (speed >= 10) & (speed < 40),
        "s40+": valid & (speed >= 40),
        "s60+": valid & (speed >= 60),
        "disc": valid & disc,
        "untex": valid & untex,
        "blur": valid & blur,
    }


def epe_by_region(pred: np.ndarray, gt: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    epe = np.linalg.norm(pred - gt, axis=-1)
    scores: dict[str, float] = {}
    for name in REGION_ORDER:
        m = masks[name]
        scores[name] = float(epe[m].mean()) if m.any() else float("nan")
    return scores


def cached_prediction(cache_root: Path, model, pair, pass_: str, no_cache: bool) -> np.ndarray:
    path = cache_root / "predictions" / model.name / "sintel" / pass_ / pair.seq / f"frame_{pair.idx:04d}.npy"
    if path.exists():
        return np.load(path)
    pred = model.predict(pair.img1, pair.img2)
    if not no_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, pred)
    return pred


def make_panel(
    out_path: Path,
    pair,
    flows: dict[str, np.ndarray],
    scores: dict[str, dict[str, float]],
) -> None:
    gt_mag = np.linalg.norm(pair.gt_flow, axis=-1)
    flow_clip = float(np.percentile(gt_mag, 99))
    epes = {name: np.linalg.norm(flow - pair.gt_flow, axis=-1) for name, flow in flows.items()}
    _, epe_vmax = normalize_heat(np.maximum.reduce(list(epes.values())))
    epe_rgb = {name: heat_to_rgb(np.clip(epe / epe_vmax, 0, 1)) for name, epe in epes.items()}
    rel_basic_rgb, rel_basic_vmax = relative_error_to_rgb(epes["gmflow_basic"] - epes["raft"])
    rel_refine_rgb = None
    rel_refine_vmax = None
    if "gmflow_refine" in epes:
        rel_refine_rgb, rel_refine_vmax = relative_error_to_rgb(epes["gmflow_refine"] - epes["raft"])

    masks = masks_for_pair(pair)
    overlay = mask_overlay(pair.img1, masks)

    score_names = [("raft", "R"), ("gmflow_basic", "B")]
    if "gmflow_refine" in scores:
        score_names.append(("gmflow_refine", "F"))

    def score_line(region: str) -> str:
        vals = " / ".join(
            f"{short} {scores[name][region]:.2f}" for name, short in score_names
        )
        return f"{region:<9} {vals}"

    score_label = " / ".join(short for _, short in score_names)
    lines = [
        f"{pair.seq}:{pair.idx:04d}",
        f"Region EPE px: {score_label}",
        score_line("all"),
        score_line("unmatched"),
        score_line("s40+"),
        score_line("s60+"),
        score_line("disc"),
        score_line("untex"),
        score_line("blur"),
        "Overlay: red occ, yellow disc",
        "blue untex, purple fast, green blur",
        "Rel: red GMFlow worse, blue RAFT worse",
        "R=RAFT, B=basic, F=refine",
    ]
    tiles = [
        tile("Input", pair.img1),
        tile("GT flow", flow_to_rgb(pair.gt_flow, flow_clip)),
        tile("Regions", overlay),
        text_tile(lines),
        tile("RAFT flow", flow_to_rgb(flows["raft"], flow_clip)),
        tile("GMFlow basic flow", flow_to_rgb(flows["gmflow_basic"], flow_clip)),
    ]
    if "gmflow_refine" in flows:
        tiles.append(tile("GMFlow refine flow", flow_to_rgb(flows["gmflow_refine"], flow_clip)))
    tiles.extend([
        tile(f"RAFT EPE (vmax {epe_vmax:.1f}px)", epe_rgb["raft"]),
        tile(f"GMFlow basic EPE (vmax {epe_vmax:.1f}px)", epe_rgb["gmflow_basic"]),
    ])
    if "gmflow_refine" in epe_rgb:
        tiles.append(tile(f"GMFlow refine EPE (vmax {epe_vmax:.1f}px)", epe_rgb["gmflow_refine"]))
    tiles.append(tile(f"Basic EPE - RAFT EPE (±{rel_basic_vmax:.1f}px)", rel_basic_rgb))
    if rel_refine_rgb is not None:
        tiles.append(tile(f"Refine EPE - RAFT EPE (±{rel_refine_vmax:.1f}px)", rel_refine_rgb))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_image(tiles, cols=4).save(out_path)


def make_region_chart(out_path: Path, rows: list[dict[str, str | float]]) -> None:
    models = list(dict.fromkeys(str(r["model"]) for r in rows))
    vals = {model: [] for model in models}
    for region in REGION_ORDER:
        for model in vals:
            numbers = [
                float(r["epe"])
                for r in rows
                if r["model"] == model and r["region"] == region and np.isfinite(float(r["epe"]))
            ]
            vals[model].append(float(np.mean(numbers)) if numbers else float("nan"))

    w, h = 1250, 640
    margin_l, margin_r, margin_t, margin_b = 90, 35, 60, 110
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    max_v = max([v for model_vals in vals.values() for v in model_vals if np.isfinite(v)] + [1.0])

    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f = font(18)
    small = font(15)
    d.text((margin_l, 18), "Region EPE on selected examples", fill=(20, 20, 20), font=font(24))
    d.text((margin_l, 46), "Lower is better. RAFT = blue, GMFlow basic = orange, refine = green.", fill=(70, 70, 70), font=small)

    for i in range(6):
        y = margin_t + plot_h - round(plot_h * i / 5)
        v = max_v * i / 5
        d.line((margin_l, y, margin_l + plot_w, y), fill=(225, 225, 225))
        d.text((14, y - 9), f"{v:.1f}", fill=(70, 70, 70), font=small)
    d.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill=(40, 40, 40))
    d.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill=(40, 40, 40))

    group_w = plot_w / len(REGION_ORDER)
    bar_w = group_w * (0.7 / max(len(models), 1))
    colors = {
        "raft": (55, 110, 210),
        "gmflow_basic": (230, 130, 45),
        "gmflow_refine": (55, 160, 95),
    }
    for i, region in enumerate(REGION_ORDER):
        cx = margin_l + i * group_w + group_w / 2
        for j, model_name in enumerate(models):
            offset = (j - (len(models) - 1) / 2) * bar_w * 1.15
            v = vals[model_name][i]
            if not np.isfinite(v):
                continue
            bh = plot_h * v / max_v
            x0 = round(cx + offset - bar_w / 2)
            x1 = round(cx + offset + bar_w / 2)
            y0 = round(margin_t + plot_h - bh)
            y1 = margin_t + plot_h
            d.rectangle((x0, y0, x1, y1), fill=colors.get(model_name, (90, 90, 90)))
        d.text((round(cx - group_w * 0.38), margin_t + plot_h + 16), region, fill=(30, 30, 30), font=small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default="datasets/Sintel")
    ap.add_argument("--pass", dest="pass_", choices=["clean", "final"], default="clean")
    ap.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES), help="Sequence/frame pairs like market_2:20")
    ap.add_argument("--out", default="results/figures/paper")
    ap.add_argument("--cache-root", default="results")
    ap.add_argument("--raft-ckpt", default="RAFT/RAFT/models/raft-things.pth")
    ap.add_argument("--gmflow-ckpt", default="gmflow/gmflow/pretrained/gmflow_things-e9887eda.pth")
    ap.add_argument("--gmflow-refine-ckpt", default="gmflow/gmflow/pretrained/gmflow_with_refine_things-36579974.pth")
    ap.add_argument("--no-refine", action="store_true", help="Only plot RAFT vs GMFlow-basic.")
    ap.add_argument("--raft-iters", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    cache_root = Path(args.cache_root)
    samples = parse_samples(args.samples)

    print("loading RAFT...")
    from cvflow.models.raft_wrapper import RaftWrapper
    raft = RaftWrapper(args.raft_ckpt, iters=args.raft_iters, device=args.device)

    print("loading GMFlow...")
    from cvflow.models.gmflow_wrapper import GMFlowWrapper
    gmflow_basic = GMFlowWrapper(args.gmflow_ckpt, device=args.device)
    gmflow_refine = None
    if not args.no_refine:
        print("loading GMFlow refine...")
        gmflow_refine = GMFlowWrapper(args.gmflow_refine_ckpt, device=args.device, **GMFLOW_REFINE_PRESET)

    rows: list[dict[str, str | float]] = []
    for seq, idx in samples:
        print(f"sample {seq}:{idx:04d}")
        pair = get_pair(args.sintel_root, args.pass_, seq, idx)
        masks = masks_for_pair(pair)
        flow_map = {
            "raft": cached_prediction(cache_root, raft, pair, args.pass_, args.no_cache),
            "gmflow_basic": cached_prediction(cache_root, gmflow_basic, pair, args.pass_, args.no_cache),
        }
        if gmflow_refine is not None:
            flow_map["gmflow_refine"] = cached_prediction(cache_root, gmflow_refine, pair, args.pass_, args.no_cache)
        score_map = {name: epe_by_region(flow, pair.gt_flow, masks) for name, flow in flow_map.items()}
        for model_name, scores in score_map.items():
            for region, epe in scores.items():
                rows.append({"seq": seq, "idx": idx, "model": model_name, "region": region, "epe": epe})
        make_panel(out / "panels" / f"{args.pass_}_{seq}_{idx:04d}.png", pair, flow_map, score_map)

    csv_path = out / f"{args.pass_}_region_scores.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq", "idx", "model", "region", "epe"])
        writer.writeheader()
        writer.writerows(rows)
    make_region_chart(out / f"{args.pass_}_region_scores.png", rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
