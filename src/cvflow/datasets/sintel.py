from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from cvflow.flow_io import read_flo, read_image, read_mask_png


@dataclass
class SintelPair:
    seq: str
    idx: int                  # 1-based; the pair is frame_{idx:04d} -> frame_{idx+1:04d}
    img1: np.ndarray          # uint8 HxWx3
    img2: np.ndarray
    gt_flow: np.ndarray       # float32 HxWx2
    occlusion: np.ndarray     # uint8 HxW, 255 = occluded
    invalid: np.ndarray       # uint8 HxW, 255 = unreliable GT


class Sintel:
    def __init__(self, root: str | Path, split: str = "training", pass_: str = "clean"):
        self.root = Path(root) / split
        if pass_ not in ("clean", "final"):
            raise ValueError(f"pass_ must be 'clean' or 'final', got {pass_!r}")
        self.pass_ = pass_
        self.seqs = sorted(p.name for p in (self.root / pass_).iterdir() if p.is_dir())

    def pairs(self, seqs: list[str] | None = None) -> Iterator[SintelPair]:
        for seq in (seqs or self.seqs):
            img_dir = self.root / self.pass_ / seq
            flow_dir = self.root / "flow" / seq
            occ_dir = self.root / "occlusions" / seq
            inv_dir = self.root / "invalid" / seq
            frames = sorted(img_dir.glob("frame_*.png"))
            for k in range(len(frames) - 1):
                idx = k + 1
                yield SintelPair(
                    seq=seq,
                    idx=idx,
                    img1=read_image(frames[k]),
                    img2=read_image(frames[k + 1]),
                    gt_flow=read_flo(flow_dir / f"frame_{idx:04d}.flo"),
                    occlusion=read_mask_png(occ_dir / f"frame_{idx:04d}.png"),
                    invalid=read_mask_png(inv_dir / f"frame_{idx:04d}.png"),
                )

    def count(self) -> int:
        return sum(len(list((self.root / self.pass_ / s).glob("frame_*.png"))) - 1 for s in self.seqs)
