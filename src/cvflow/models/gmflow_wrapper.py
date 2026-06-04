from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from cvflow.models._padder import InputPadder

_GMFLOW_ROOT = Path(__file__).resolve().parents[3] / "gmflow" / "gmflow"


def _import_gmflow():
    # gmflow.gmflow uses only relative imports internally — no `utils.utils`
    # dependency — but we still clear the gmflow.* cache so RAFT-loading
    # order doesn't matter.
    for k in [k for k in sys.modules if k == "gmflow" or k.startswith("gmflow.")]:
        del sys.modules[k]
    if str(_GMFLOW_ROOT) not in sys.path:
        sys.path.insert(0, str(_GMFLOW_ROOT))
    from gmflow.gmflow import GMFlow  # type: ignore[import-not-found]
    return GMFlow


class GMFlowWrapper:
    """GMFlow inference. Defaults match the basic (no-refinement) preset from
    gmflow/scripts/evaluate.sh; the with-refinement preset takes
        num_scales=2, upsample_factor=4, padding_factor=32,
        attn_splits_list=[2, 8], corr_radius_list=[-1, 4], prop_radius_list=[-1, 1].
    """

    def __init__(
        self,
        checkpoint: str | Path,
        padding_factor: int = 16,
        attn_splits_list: list[int] | int = (2,),
        corr_radius_list: list[int] | int = (-1,),
        prop_radius_list: list[int] | int = (-1,),
        num_scales: int = 1,
        upsample_factor: int = 8,
        device: str | torch.device | None = None,
    ):
        def _aslist(x):
            return list(x) if hasattr(x, "__iter__") else [x]

        self.padding_factor = padding_factor
        self.attn_splits_list = _aslist(attn_splits_list)
        self.corr_radius_list = _aslist(corr_radius_list)
        self.prop_radius_list = _aslist(prop_radius_list)
        assert len(self.attn_splits_list) == len(self.corr_radius_list) == len(self.prop_radius_list) == num_scales, \
            "attn_splits_list / corr_radius_list / prop_radius_list lengths must equal num_scales"

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"gmflow-{Path(checkpoint).stem}"

        GMFlow = _import_gmflow()

        model = GMFlow(
            num_scales=num_scales,
            upsample_factor=upsample_factor,
            feature_channels=128,
            attention_type="swin",
            num_transformer_layers=6,
            ffn_dim_expansion=4,
            num_head=1,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(sd, strict=True)
        model.to(self.device).eval()
        self.model = model

    @torch.no_grad()
    def predict(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        t1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        t2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        padder = InputPadder(t1.shape, padding_factor=self.padding_factor)
        t1p, t2p = padder.pad(t1, t2)
        out = self.model(
            t1p, t2p,
            attn_splits_list=self.attn_splits_list,
            corr_radius_list=self.corr_radius_list,
            prop_radius_list=self.prop_radius_list,
            pred_bidir_flow=False,
        )
        flow_up = out["flow_preds"][-1]
        flow = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy()
        return flow.astype(np.float32)
