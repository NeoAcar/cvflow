from __future__ import annotations

import torch.nn.functional as F


class InputPadder:
    """Pads a 4D tensor's H,W to multiples of `padding_factor`.

    Same arithmetic as the InputPadder in both RAFT and GMFlow source trees,
    inlined here to avoid the `utils.utils` namespace collision between the
    two cloned repos (RAFT ships `utils/` as a package, GMFlow ships it as a
    bare directory without `__init__.py`).
    """

    def __init__(self, dims, mode: str = "sintel", padding_factor: int = 8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:  # kitti / asymmetric bottom-padding
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
