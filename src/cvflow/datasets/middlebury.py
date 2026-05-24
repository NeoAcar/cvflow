from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from cvflow.flow_io import read_flo, read_image


@dataclass
class MiddleburyPair:
    seq: str
    img1: np.ndarray          # uint8 HxWx3
    img2: np.ndarray
    gt_flow: np.ndarray       # float32 HxWx2


class Middlebury:
    """Middlebury 'other' set, restricted to sequences with GT flow."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        gt_dir = self.root / "other-gt-flow"
        self.seqs = sorted(p.name for p in gt_dir.iterdir() if p.is_dir() and (p / "flow10.flo").exists())

    def pairs(self) -> Iterator[MiddleburyPair]:
        for seq in self.seqs:
            img_dir = self.root / "other-data" / seq
            yield MiddleburyPair(
                seq=seq,
                img1=read_image(img_dir / "frame10.png"),
                img2=read_image(img_dir / "frame11.png"),
                gt_flow=read_flo(self.root / "other-gt-flow" / seq / "flow10.flo"),
            )
