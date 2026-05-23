from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from cvflow.models._padder import InputPadder

_RAFT_CORE = Path(__file__).resolve().parents[3] / "RAFT" / "RAFT" / "core"


def _import_raft():
    # raft.py itself does `from utils.utils import bilinear_sampler, ...`, so
    # we must ensure that resolves to RAFT/core/utils/utils.py — not to GMFlow's
    # `utils/` (which has no __init__.py and would shadow nothing, but if
    # anything else has cached a different `utils` we clear it).
    for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[k]
    for k in ("raft", "corr", "extractor", "update"):
        sys.modules.pop(k, None)
    if str(_RAFT_CORE) not in sys.path:
        sys.path.insert(0, str(_RAFT_CORE))
    from raft import RAFT  # type: ignore[import-not-found]
    return RAFT


class RaftWrapper:
    def __init__(
        self,
        checkpoint: str | Path,
        iters: int = 32,
        small: bool = False,
        mixed_precision: bool = False,
        device: str | torch.device | None = None,
    ):
        self.iters = iters
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"raft-{Path(checkpoint).stem}-iter{iters}"

        RAFT = _import_raft()

        args = Namespace(small=small, mixed_precision=mixed_precision, alternate_corr=False, dropout=0)
        model = RAFT(args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model

    @torch.no_grad()
    def predict(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        t1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        t2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        padder = InputPadder(t1.shape, mode="sintel", padding_factor=8)
        t1p, t2p = padder.pad(t1, t2)
        _, flow_up = self.model(t1p, t2p, iters=self.iters, test_mode=True)
        flow = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy()
        return flow.astype(np.float32)
