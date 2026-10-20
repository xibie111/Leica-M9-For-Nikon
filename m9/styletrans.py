"""StyleTrans node (com.xiaomi.plugin.legendST -> libmialgo_styletrans.so), reimplemented around
a pluggable network backend. Pre/post-processing follows styletrans_pipeline.cpp run() 0x449518
(see M9-style-analysis.md):

  rgb (B2Y NV12 decoded, BT.601 full) -> centre pad to 4096x3072 (REFLECT_101)
  -> INTER_AREA 1024x768 -> [mask from humanseg] -> pad 48 (RGB REFLECT, mask 0) -> 1120x864x4
  -> u8 graph input (net sees (x-128)/127.5) -> styletrans_{low,high} -> crop Rect(48,48,1024,768)
  -> INTER_AREA up to 4096x3072 -> colorfix 544x544x6 tiles (step 512, centre 512 kept)
  -> centre crop back.
Model choice: low if lux_index > 260 and CCT < 4692 else high.

Backends: see `load()`. A backend is an object with
    run(model: str, x: np.ndarray[N,H,W,C] float32) -> np.ndarray[N,H,W,3] float32 (0..255 scale)
"""
from __future__ import annotations

import os

import cv2
import numpy as np

W4K, H4K = 4096, 3072
NW, NH = 1024, 768
PAD = 48
TILE, STEP, MARGIN = 544, 512, 16


def choose_model(lux_index: float, cct: float) -> str:
    return "low" if (lux_index > 260 and cct < 4692) else "high"


def _pad_center(img, w, h, border):
    dh, dw = h - img.shape[0], w - img.shape[1]
    t, l = dh // 2, dw // 2
    return cv2.copyMakeBorder(img, t, dh - t, l, dw - l, border), (t, l)


def _quantize_style_input(x):
    """Quantize RGB with an owned output and at most 128 rows of FP32 scratch."""
    if x.dtype != np.float32 or 1 in x.shape:
        return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out = np.empty_like(x, dtype=np.uint8, order="K")
    scratch = np.empty_like(x[:min(128, x.shape[0])], dtype=np.float32, order="K")
    for row in range(0, x.shape[0], 128):
        count = min(128, x.shape[0] - row)
        work = scratch[:count]
        np.multiply(x[row:row + count], np.float32(255.0), out=work)
        np.add(work, np.float32(0.5), out=work)
        np.clip(work, 0, 255, out=work)
        np.copyto(out[row:row + count], work, casting="unsafe")
    return out


class StyleTrans:
    def __init__(self, backend, humanseg=None):
        self.be = backend
        self.humanseg = humanseg  # callable(rgb_u8 768x1024) -> mask u8 768x1024, or None

    def __call__(self, rgb: np.ndarray, lux_index: float, cct: float, mask=None,
                 resolution="phone", progress=None) -> np.ndarray:
        """Return styled float RGB of the same size.

        ``phone`` retains the original 4096x3072 detail branch. ``original`` fits only
        the style preview to that canvas, then colorfix uses untouched full-size pixels.
        A supplied mask uses the input image's coordinates and is resized with the preview.
        """
        if resolution not in ("original", "phone"):
            raise ValueError("resolution must be original or phone")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("rgb must be a nonempty HxWx3 image")
        for y in range(0, rgb.shape[0], 256):
            if not np.isfinite(rgb[y:y + 256]).all():
                raise ValueError("StyleTrans RGB input contains NaN or infinity")
        if mask is not None and np.shape(mask) != rgb.shape[:2]:
            raise ValueError("mask must have the same height and width as rgb")
        portrait = rgb.shape[0] > rgb.shape[1]
        x = np.rot90(rgb, -1) if portrait else rgb          # sensor (landscape) orientation
        x8 = _quantize_style_input(x)
        full = x8
        h0, w0 = x8.shape[:2]
        if mask is not None:
            mask = np.rot90(mask, -1) if portrait else mask
        if h0 > H4K or w0 > W4K:                            # offline: fit the 12.5 MP M9 frame
            s = min(W4K / w0, H4K / h0)
            size = (max(1, int(w0 * s)), max(1, int(h0 * s)))
            x8 = cv2.resize(x8, size, interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        h1, w1 = x8.shape[:2]
        big, (t, l) = _pad_center(x8, W4K, H4K, cv2.BORDER_REFLECT_101)

        small = cv2.resize(big, (NW, NH), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask, _ = _pad_center(mask, W4K, H4K, cv2.BORDER_CONSTANT)
            mask = cv2.resize(mask, (NW, NH), interpolation=cv2.INTER_NEAREST)
        else:
            mask = self.humanseg(small) if self.humanseg else np.zeros((NH, NW), np.uint8)
        rgbp = cv2.copyMakeBorder(small, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT)
        mp = cv2.copyMakeBorder(mask, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
        # graph input is UFIXED8 (scale 1/127.5, offset -128): raw pixels, the (x-128)/127.5
        # normalisation of the library lives in the quantisation parameters
        inp = np.concatenate([rgbp, mp[..., None]], -1)
        model = choose_model(lux_index, cct)
        if progress:
            progress("style_network", f"StyleTrans {model}: {NW}x{NH} preview")
        out = self._run_bytes(model, inp[None])[0]          # 864x1120x3, 0..255
        if out.shape != (NH + 2 * PAD, NW + 2 * PAD, 3):
            raise RuntimeError(f"StyleTrans returned unexpected shape {out.shape}")
        if not np.isfinite(out).all():
            raise RuntimeError("StyleTrans network output contains NaN or infinity")
        styled = np.clip(out[PAD:PAD + NH, PAD:PAD + NW], 0, 255).astype(np.uint8)
        up = cv2.resize(styled, (W4K, H4K), interpolation=cv2.INTER_AREA)

        if resolution == "original":
            # Remove the preview padding BEFORE scaling: padding is not part of the photograph.
            mapped = up[t:t + h1, l:l + w1]
            if (h1, w1) != (h0, w0):
                mapped = cv2.resize(mapped, (w0, h0), interpolation=cv2.INTER_LINEAR)
            res = self._colorfix(full, mapped, progress)
        else:
            fixed = self._colorfix(big, up, progress)
            res = fixed[t:t + h1, l:l + w1]
            if (h1, w1) != (h0, w0):
                res = cv2.resize(res, (w0, h0), interpolation=cv2.INTER_LINEAR)
        res = res.astype(np.float32) / 255.0
        return np.rot90(res, 1) if portrait else res

    def _run_bytes(self, model, pixels):
        run_u8 = getattr(self.be, "run_u8", None)
        if run_u8 is not None and pixels.dtype == np.uint8:
            return run_u8(model, pixels)
        return self.be.run(model, pixels.astype(np.float32))

    def _colorfix(self, orig, styled, progress=None):
        H, W = orig.shape[:2]
        out = np.empty_like(orig)
        total = ((H + STEP - 1) // STEP) * ((W + STEP - 1) // STEP)
        done = 0
        t = np.empty((TILE, TILE, 6), np.uint8)
        for y in range(0, H, STEP):
            for x in range(0, W, STEP):
                ys, xs = max(y - MARGIN, 0), max(x - MARGIN, 0)
                ye, xe = min(y + STEP + MARGIN, H), min(x + STEP + MARGIN, W)
                oy, ox = MARGIN - (y - ys), MARGIN - (x - xs)
                t.fill(0)
                t[oy:oy + ye - ys, ox:ox + xe - xs, :3] = styled[ys:ye, xs:xe]
                t[oy:oy + ye - ys, ox:ox + xe - xs, 3:] = orig[ys:ye, xs:xe]
                r = self._run_bytes("colorfix", t[None])
                if r.shape != (1, TILE, TILE, 3):
                    raise RuntimeError(f"Colorfix returned unexpected shape {r.shape}")
                if not np.isfinite(r).all():
                    raise RuntimeError("Colorfix network output contains NaN or infinity")
                h, w = min(STEP, H - y), min(STEP, W - x)
                center = r[0, MARGIN:MARGIN + h, MARGIN:MARGIN + w]
                out[y:y + h, x:x + w] = center if r.dtype == np.uint8 else np.clip(center, 0, 255)
                done += 1
            if progress:
                progress("colorfix", f"Colorfix {done}/{total} tiles ({W}x{H})")
        return out


def load(device=None, strict=False):
    """Load StyleTrans; strict processing requires all three PyTorch networks.

    The legacy DNG renderer keeps its optional backend behavior by default.
    """
    be = None
    if strict and os.environ.get("M9_BACKEND", "torch") != "torch":
        raise RuntimeError("Strict StyleTrans requires the torch backend (M9_BACKEND=torch)")
    if os.environ.get("M9_BACKEND", "torch") == "torch":
        try:
            from .torch_backend import TorchBackend
            be = TorchBackend.create(device=device, strict=strict)
        except ImportError as error:
            if strict:
                raise RuntimeError("StyleTrans requires the installed PyTorch backend") from error
            be = None
    if be is None and os.environ.get("M9_BACKEND") == "qnn":   # simulator: opt-in only (minutes/image)
        try:
            from .qnn_backend import QnnSimBackend
            be = QnnSimBackend.create()
        except Exception:  # noqa: BLE001 - backend optional
            be = None
    if be is None and strict:
        raise RuntimeError("StyleTrans backend unavailable; processing cannot continue")
    return StyleTrans(be) if be is not None else None
