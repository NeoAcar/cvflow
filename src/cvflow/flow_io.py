from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_FLO_MAGIC = b"PIEH"


def read_flo(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != _FLO_MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}, expected {_FLO_MAGIC!r}")
        w = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        h = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        if not (1 <= w <= 100000 and 1 <= h <= 100000):
            raise ValueError(f"{path}: implausible dims {w}x{h}")
        data = np.frombuffer(f.read(2 * w * h * 4), dtype=np.float32)
    return data.reshape(h, w, 2).copy()


def read_mask_png(path: str | Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def read_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
