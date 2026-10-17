"""Fast StyleTrans backend: float PyTorch networks rebuilt from the weights decoded out of the
ROM context binaries (styletrans_net.py / colorfix_net.py, weights re/weights/*.npz).

Same I/O contract as QnnSimBackend: u8-domain pixels in, u8-domain pixels out. The graph I/O is
UFIXED8 scale 1/127.5 offset -128, so the net sees (x-128)/127.5 and its [-1,1] output is
re-quantised as round(y*127.5+128).
Config: M9_WEIGHTS (dir with low/high/colorfix.npz), M9_DEVICE (cuda/cpu).
"""
from __future__ import annotations

import os

import numpy as np

from .device import resolve_device

_HERE = os.path.dirname(os.path.abspath(__file__))
DEF_WEIGHTS = os.path.join(_HERE, "..", "re", "weights")


class TorchBackend:
    def __init__(self, wdir: str, device: str):
        self.wdir, self.device, self.nets = wdir, device, {}

    @classmethod
    def create(cls, device=None, strict=False):
        import torch
        wdir = os.environ.get("M9_WEIGHTS", DEF_WEIGHTS)
        missing = [m for m in ("low", "high", "colorfix")
                   if not os.path.isfile(os.path.join(wdir, f"{m}.npz"))]
        if missing:
            if strict:
                raise FileNotFoundError(f"StyleTrans weights missing in {wdir}: {', '.join(missing)}")
            return None
        be = cls(wdir, resolve_device(device))
        if strict:
            for model in ("low", "high", "colorfix"):
                try:
                    net = be._net(model)
                    be._check_layers(model, net)
                except Exception as error:
                    raise RuntimeError(f"Cannot load {model} weights from {wdir}: {error}") from error
        return be

    @staticmethod
    def _check_layers(model, net):
        """A readable but incomplete archive must fail during model preflight."""
        import torch
        if model == "colorfix":
            required = {"layer1_net_0", "layer1_net_3", "layer10", "layer11", "layer12"}
            required.update(f"layer{i}_net_1_net_{j}" for i in range(2, 6) for j in (0, 3))
            required.update(f"layer{i}_conv_net_{j}" for i in range(6, 10) for j in (0, 3))
            required.update(f"layer{i}_upsample" for i in range(6, 10))
        else:
            required = {"conv_first", "conv_body", "fixed_conv_conv", "tail_conv_0",
                        "tail_conv_1", "tail_conv_2", "tail_conv_last"}
            required.update(f"body_{i}_{suffix}" for i in range(20)
                            for suffix in ("conv1", "conv2", "fixed_conv_conv"))
            required.update(f"body_{i}_rdb{j}_{suffix}" for i in range(20) for j in (1, 2, 3)
                            for suffix in ("block1_conv1", "block1_conv2", "block2_conv1",
                                           "block2_conv2", "conv", "fixed_conv_conv", "conv_out"))
        missing = required.difference(net.p)
        if missing:
            raise ValueError(f"Missing network layers: {', '.join(sorted(missing)[:5])}")
        finite = []
        for name in required:
            weights, bias, lo, hi = net.p[name]
            if (weights.ndim != 4 or bias.ndim != 1 or weights.shape[0] != bias.shape[0]
                    or not np.isfinite([lo, hi]).all() or lo > hi):
                raise ValueError(f"Invalid convolution weights: {name}")
            finite.append(torch.isfinite(weights).all() & torch.isfinite(bias).all())
        if not torch.stack(finite).all().item():
            raise ValueError("Network weights or bias contain NaN or infinity")

    def _net(self, model):
        if model not in self.nets:
            p = os.path.join(self.wdir, f"{model}.npz")
            if model == "colorfix":
                from .colorfix_net import ColorFixNet
                self.nets[model] = ColorFixNet(p, device=self.device)
            else:
                from .styletrans_net import StyleTransNet
                self.nets[model] = StyleTransNet(p, device=self.device)
        return self.nets[model]

    def run(self, model: str, x: np.ndarray) -> np.ndarray:
        import torch
        if not np.isfinite(x).all():
            raise ValueError(f"{model} network input contains NaN or infinity")
        net = self._net(model)
        q = (np.clip(np.rint(x), 0, 255).astype(np.float32) - 128.0) / 127.5
        outs = []
        # Colorfix is also called one tile at a time; keep that bound for direct callers.
        bs = 1
        with torch.inference_mode():
            for i in range(0, len(q), bs):
                t = torch.from_numpy(q[i:i + bs]).permute(0, 3, 1, 2).to(self.device)
                y = net(t) if model == "colorfix" else net(t[:, :3], t[:, 3:4])
                if not torch.isfinite(y).all().item():
                    raise RuntimeError(f"{model} network output contains NaN or infinity")
                outs.append(y.permute(0, 2, 3, 1).cpu().numpy())
        y = np.concatenate(outs)
        return np.clip(np.rint(y * 127.5 + 128.0), 0, 255).astype(np.float32)
