import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from m9 import styletrans
from m9.device import resolve_device
from m9.torch_backend import DEF_WEIGHTS, TorchBackend


class IdentityBackend:
    def __init__(self, detail=True):
        self.detail = detail
        self.calls = []

    def run(self, model, image):
        self.calls.append((model, image.shape))
        if model == "colorfix":
            assert image.shape == (1, 544, 544, 6)
            return image[..., 3:] if self.detail else image[..., :3]
        return image[..., :3]


def compact_canvas(monkeypatch):
    for key, value in dict(W4K=64, H4K=48, NW=16, NH=12, PAD=4).items():
        monkeypatch.setattr(styletrans, key, value)


@pytest.mark.parametrize("portrait", [False, True])
def test_original_detail_is_not_downsampled(portrait):
    # Wider than the actual phone canvas, with one-pixel detail lost by a resize.
    y, x = np.indices((529, 4301))
    u8 = np.stack([(x % 2) * 255, (y % 2) * 255, (x + 3 * y) % 256], -1).astype(np.uint8)
    if portrait:
        u8 = np.rot90(u8)
    backend = IdentityBackend()
    result = styletrans.StyleTrans(backend)(u8.astype(np.float32) / 255, 0, 5000,
                                          resolution="original")
    np.testing.assert_array_equal(np.rint(result * 255).astype(np.uint8), u8)
    assert len([c for c in backend.calls if c[0] == "colorfix"]) == 2 * 9
    assert backend.calls[0] == ("high", (1, 864, 1120, 4))


@pytest.mark.parametrize("shape", [(1, 1), (513, 517), (544, 1025)])
def test_colorfix_single_tile_batches_and_edge_centers(shape):
    original = np.random.default_rng(10).integers(0, 256, (*shape, 3), dtype=np.uint8)
    backend = IdentityBackend()
    result = styletrans.StyleTrans(backend)._colorfix(original, np.zeros_like(original))
    np.testing.assert_array_equal(result, original)
    expected = ((shape[0] + 511) // 512) * ((shape[1] + 511) // 512)
    assert len(backend.calls) == expected
    assert all(size == (1, 544, 544, 6) for _, size in backend.calls)


@pytest.mark.parametrize("shape", [(70, 105), (105, 70), (17, 43), (43, 17), (49, 101)])
def test_original_style_mapping_removes_phone_padding_before_resize(monkeypatch, shape):
    compact_canvas(monkeypatch)
    h, w = shape
    rgb = np.random.default_rng(2).random((h, w, 3), dtype=np.float32)
    captured = {}
    st = styletrans.StyleTrans(IdentityBackend(detail=False))

    def colorfix(original, mapped, progress=None):
        captured.update(original=original.copy(), mapped=mapped.copy())
        return mapped

    monkeypatch.setattr(st, "_colorfix", colorfix)
    result = st(rgb, 300, 4000, resolution="original")
    portrait = h > w
    original = np.clip((np.rot90(rgb, -1) if portrait else rgb) * 255 + .5, 0, 255).astype(np.uint8)
    h0, w0 = original.shape[:2]
    scale = min(64 / w0, 48 / h0, 1)
    width, height = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    fitted = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA) if scale < 1 else original
    top, left = (48 - height) // 2, (64 - width) // 2
    canvas = cv2.copyMakeBorder(fitted, top, 48 - height - top, left, 64 - width - left,
                                cv2.BORDER_REFLECT_101)
    preview = cv2.resize(canvas, (16, 12), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(preview, (64, 48), interpolation=cv2.INTER_AREA)
    expected = restored[top:top + height, left:left + width]
    if scale < 1:
        expected = cv2.resize(expected, (w0, h0), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(captured["original"], original)
    np.testing.assert_array_equal(captured["mapped"], expected)
    expected = np.rot90(expected) if portrait else expected
    np.testing.assert_allclose(result, expected.astype(np.float32) / 255, atol=0)


def test_default_phone_retains_legacy_fit_and_upscale(monkeypatch):
    compact_canvas(monkeypatch)
    rgb = np.random.default_rng(4).random((71, 109, 3), dtype=np.float32)
    backend = IdentityBackend()
    result = styletrans.StyleTrans(backend)(rgb, 0, 5000)
    original = np.clip(rgb * 255 + .5, 0, 255).astype(np.uint8)
    scale = min(64 / 109, 48 / 71)
    expected = cv2.resize(original, (int(109 * scale), int(71 * scale)), interpolation=cv2.INTER_AREA)
    expected = cv2.resize(expected, (109, 71), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(result, expected.astype(np.float32) / 255, atol=0)


def test_invalid_resolution_mask_and_backend_shapes(monkeypatch):
    compact_canvas(monkeypatch)
    st = styletrans.StyleTrans(IdentityBackend())
    image = np.zeros((20, 30, 3), np.float32)
    with pytest.raises(ValueError, match="resolution"):
        st(image, 0, 5000, resolution="half")
    with pytest.raises(ValueError, match="mask"):
        st(image, 0, 5000, mask=np.zeros((1, 1)))
    st.be = SimpleNamespace(run=lambda *args: np.zeros((1, 1, 1, 3)))
    with pytest.raises(RuntimeError, match="unexpected shape"):
        st(image, 0, 5000)


def test_strict_load_cannot_use_preview_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("M9_WEIGHTS", str(tmp_path))
    monkeypatch.setenv("M9_BACKEND", "torch")
    pytest.importorskip("torch")
    assert styletrans.load(device="cpu") is None
    with pytest.raises(FileNotFoundError, match="low, high, colorfix"):
        styletrans.load(device="cpu", strict=True)
    monkeypatch.setenv("M9_BACKEND", "qnn")
    with pytest.raises(RuntimeError, match="torch backend"):
        styletrans.load(device="cpu", strict=True)


def test_strict_load_preloads_three_networks(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setenv("M9_WEIGHTS", str(tmp_path))
    for model in ("low", "high", "colorfix"):
        (tmp_path / f"{model}.npz").touch()
    loaded = []
    monkeypatch.setattr(TorchBackend, "_net", lambda self, model: loaded.append(model))
    monkeypatch.setattr(TorchBackend, "_check_layers", lambda *args: None)
    result = TorchBackend.create(device="cpu", strict=True)
    assert result.device == "cpu"
    assert loaded == ["low", "high", "colorfix"]


def test_corrupt_weights_fail_before_processing(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setenv("M9_WEIGHTS", str(tmp_path))
    monkeypatch.setenv("M9_BACKEND", "torch")
    for model in ("low", "high", "colorfix"):
        (tmp_path / f"{model}.npz").write_bytes(b"damaged npz")
    with pytest.raises(RuntimeError, match="Cannot load low weights"):
        styletrans.load(device="cpu", strict=True)


def test_readable_but_empty_weights_fail_preflight(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setenv("M9_WEIGHTS", str(tmp_path))
    monkeypatch.setenv("M9_BACKEND", "torch")
    for model in ("low", "high", "colorfix"):
        np.savez(tmp_path / f"{model}.npz")
    with pytest.raises(RuntimeError, match="Missing network layers"):
        styletrans.load(device="cpu", strict=True)


def test_device_selection_does_not_fall_back_from_explicit_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="device"):
        resolve_device("metal")


@pytest.mark.parametrize("fail_stage", ["high", "colorfix"])
def test_out_of_memory_propagates_without_retry_or_resolution_change(monkeypatch, fail_stage):
    compact_canvas(monkeypatch)
    backend = IdentityBackend()
    ordinary_run = backend.run

    def run(model, image):
        if model == fail_stage:
            backend.calls.append((model, image.shape))
            raise RuntimeError("CUDA out of memory (simulated)")
        return ordinary_run(model, image)

    backend.run = run
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        styletrans.StyleTrans(backend)(np.zeros((71, 109, 3), np.float32), 0, 5000,
                                      resolution="original")
    assert [model for model, _ in backend.calls].count(fail_stage) == 1
    assert [model for model, _ in backend.calls] == (["high"] if fail_stage == "high"
                                                   else ["high", "colorfix"])


def test_strict_load_reports_missing_torch(monkeypatch):
    monkeypatch.setenv("M9_BACKEND", "torch")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="requires the installed PyTorch"):
        styletrans.load(device="cpu", strict=True)


def test_loading_cuda_oom_is_not_hidden(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setenv("M9_BACKEND", "torch")
    monkeypatch.setenv("M9_WEIGHTS", str(tmp_path))
    for model in ("low", "high", "colorfix"):
        (tmp_path / f"{model}.npz").touch()
    called = []

    def net(self, model):
        called.append(model)
        raise RuntimeError("CUDA out of memory (simulated load)")

    monkeypatch.setattr(TorchBackend, "_net", net)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        styletrans.load(device="cpu", strict=True)
    assert called == ["low"]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_full_image_input_fails_before_quantization(bad_value):
    image = np.zeros((529, 5, 3), np.float32)
    image[-1, -1, -1] = bad_value
    backend = IdentityBackend()
    with pytest.raises(ValueError, match="RGB input contains NaN or infinity"):
        styletrans.StyleTrans(backend)(image, 0, 5000, resolution="original")
    assert backend.calls == []


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
@pytest.mark.parametrize("fail_stage", ["high", "colorfix"])
def test_nonfinite_backend_outputs_fail_before_uint8_conversion(monkeypatch, bad_value, fail_stage):
    compact_canvas(monkeypatch)
    backend = IdentityBackend()
    ordinary_run = backend.run

    def run(model, image):
        result = ordinary_run(model, image).copy()
        if model == fail_stage:
            result[0, 0, 0, 0] = bad_value
        return result

    backend.run = run
    with pytest.raises(RuntimeError, match="network output contains NaN or infinity"):
        styletrans.StyleTrans(backend)(np.zeros((20, 30, 3), np.float32), 0, 5000,
                                      resolution="original")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_torch_backend_rejects_nonfinite_raw_tensor_before_rounding(bad_value):
    torch = pytest.importorskip("torch")
    backend = TorchBackend("unused", "cpu")
    backend.nets["colorfix"] = lambda t: torch.full((1, 3, 2, 2), bad_value)
    with pytest.raises(RuntimeError, match="colorfix network output contains NaN or infinity"):
        backend.run("colorfix", np.zeros((1, 2, 2, 6), np.float32))


def test_torch_backend_rejects_nonfinite_input_before_model_load():
    pytest.importorskip("torch")
    backend = TorchBackend("missing weights", "cpu")
    with pytest.raises(ValueError, match="network input contains NaN or infinity"):
        backend.run("high", np.full((1, 2, 2, 4), np.nan, np.float32))


@pytest.mark.parametrize("field,bad_value", [("W", np.nan), ("b", np.inf)])
def test_numeric_corruption_in_readable_npz_fails_strict_preflight(monkeypatch, tmp_path,
                                                                field, bad_value):
    pytest.importorskip("torch")
    monkeypatch.setenv("M9_WEIGHTS", str(tmp_path))
    monkeypatch.setenv("M9_BACKEND", "torch")
    with np.load(Path(DEF_WEIGHTS) / "low.npz") as original:
        weights = {key: original[key] for key in original.files}
    key = "conv_first/" + field
    weights[key] = weights[key].copy()
    weights[key].flat[0] = bad_value
    np.savez(tmp_path / "low.npz", **weights)
    for model in ("high", "colorfix"):
        (tmp_path / f"{model}.npz").touch()
    with pytest.raises(RuntimeError, match="weights or bias contain NaN or infinity"):
        styletrans.load(device="cpu", strict=True)
