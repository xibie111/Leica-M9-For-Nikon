"""Exact byte-path contracts; CUDA checks are opt-in while GPU jobs are coordinated."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from m9.styletrans import StyleTrans
from m9.torch_backend import TorchBackend


@pytest.mark.parametrize("model,channels", [("low", 4), ("high", 4), ("colorfix", 6)])
def test_byte_cpu_fallback_matches_float_api_and_keeps_batch_one(model, channels):
    torch = pytest.importorskip("torch")
    backend = TorchBackend("unused", "cpu")
    calls = []

    def network(*parts):
        calls.append(tuple(part.shape for part in parts))
        return parts[0][:, :3]

    backend.nets[model] = network
    image = np.arange(2 * 8 * 16 * channels, dtype=np.uint16).astype(np.uint8).reshape(2, 8, 16, channels)
    frozen = image.copy()
    expected = backend.run(model, image.astype(np.float32))
    calls.clear()
    actual = backend.run_u8(model, image)
    assert actual.dtype == np.uint8 and actual.shape == (2, 8, 16, 3)
    np.testing.assert_array_equal(actual, expected.astype(np.uint8))
    np.testing.assert_array_equal(image, frozen)
    assert len(calls) == 2 and all(parts[0][0] == 1 for parts in calls)
    first = actual.copy()
    backend.run_u8(model, np.bitwise_not(image))
    np.testing.assert_array_equal(actual, first)
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_byte_cpu_fallback_rejects_nonfinite_network_output(bad):
    torch = pytest.importorskip("torch")
    backend = TorchBackend("unused", "cpu")
    backend.nets["colorfix"] = lambda pixels: torch.full((1, 3, 2, 2), bad)
    with pytest.raises(RuntimeError, match="output contains NaN or infinity"):
        backend.run_u8("colorfix", np.zeros((1, 2, 2, 6), np.uint8))


def test_nonstandard_cuda_shape_uses_legacy_api_without_gpu(monkeypatch):
    backend = TorchBackend("unused", "cuda")
    image = np.arange(2 * 3 * 4 * 6, dtype=np.uint8).reshape(2, 3, 4, 6)
    calls = []

    def run(model, pixels):
        calls.append((model, pixels.copy()))
        return pixels[..., :3]

    monkeypatch.setattr(backend, "run", run)
    actual = backend.run_u8("colorfix", image)
    assert len(calls) == 1 and calls[0][0] == "colorfix"
    assert calls[0][1].dtype == np.float32
    np.testing.assert_array_equal(calls[0][1], image)
    np.testing.assert_array_equal(actual, image[..., :3])
    assert actual.dtype == np.uint8 and not np.shares_memory(actual, image)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_byte_api_rejects_unquantized_input(dtype):
    backend = TorchBackend("missing weights", "cpu")
    with pytest.raises(TypeError, match="uint8"):
        backend.run_u8("high", np.zeros((1, 2, 3, 4), dtype))


def test_style_byte_dispatch_preserves_custom_float_backends():
    calls = []

    def run(model, pixels):
        calls.append(("float", model, pixels.dtype))
        return pixels[..., :3]

    def run_u8(model, pixels):
        calls.append(("bytes", model, pixels.dtype))
        return pixels[..., :3]

    st = StyleTrans(SimpleNamespace(run=run, run_u8=run_u8))
    byte_input = np.zeros((1, 2, 3, 4), np.uint8)
    st._run_bytes("high", byte_input)
    st._run_bytes("high", byte_input.astype(np.float32))
    st.be = SimpleNamespace(run=run)
    st._run_bytes("high", byte_input)
    assert calls == [("bytes", "high", np.dtype("uint8")),
                     ("float", "high", np.dtype("float32")),
                     ("float", "high", np.dtype("float32"))]


@pytest.mark.parametrize("shape", [(1, 1), (513, 517), (544, 1025)])
def test_reused_byte_tile_has_zero_edges_and_independent_output(shape):
    original = np.random.default_rng(26).integers(1, 256, (*shape, 3), np.uint8)
    styled = np.bitwise_not(original)
    expected_tiles = []
    for y in range(0, shape[0], 512):
        for x in range(0, shape[1], 512):
            tile = np.zeros((544, 544, 6), np.uint8)
            for row in range(544):
                source_y = y + row - 16
                if 0 <= source_y < shape[0]:
                    left, right = max(0, x - 16), min(shape[1], x + 528)
                    tile[row, left - x + 16:right - x + 16, :3] = styled[source_y, left:right]
                    tile[row, left - x + 16:right - x + 16, 3:] = original[source_y, left:right]
            expected_tiles.append(tile)
    calls, progress = [], []

    def run_u8(model, pixels):
        assert model == "colorfix" and pixels.shape == (1, 544, 544, 6)
        np.testing.assert_array_equal(pixels[0], expected_tiles[len(calls)])
        calls.append(True)
        # Returning a view exposes aliasing if the caller fails to copy each center.
        return pixels[..., 3:]

    st = StyleTrans(SimpleNamespace(run_u8=run_u8, run=lambda *args: pytest.fail("byte path unused")))
    result = st._colorfix(original, styled, lambda stage, message: progress.append((stage, message)))
    np.testing.assert_array_equal(result, original)
    assert not np.shares_memory(result, original)
    result[:] = 0
    assert np.any(original)
    assert len(calls) == len(expected_tiles)
    assert len(progress) == (shape[0] + 511) // 512


CUDA_OPT_IN = os.environ.get("M9_RUN_STRICT_SPEED_CUDA") == "1"


@pytest.mark.skipif(not CUDA_OPT_IN, reason="Explicit GPU scheduling required")
@pytest.mark.parametrize("model,shape", [("low", (1, 864, 1120, 4)), ("high", (1, 864, 1120, 4)),
                                       ("colorfix", (1, 544, 544, 6))])
def test_cuda_bytes_match_float_rounding_stride_and_return_ownership(model, shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    backend = TorchBackend("unused", "cuda")
    image = np.arange(np.prod(shape), dtype=np.uint32).astype(np.uint8).reshape(shape)
    normalized = (image.astype(np.float32) - 128.0) / 127.5
    boundaries = np.array([-2, -1, -0.5, 0, 0.5, 1, 1.5, 126.5, 127.5, 128.5, 254.5, 255, 256], np.float32)
    levels = np.concatenate([np.nextafter(boundaries, -np.inf), boundaries,
                             np.nextafter(boundaries, np.inf)])
    values = np.resize((levels - np.float32(128)) / np.float32(127.5), (1, shape[1], shape[2], 3)).copy()
    output = torch.from_numpy(values).permute(0, 3, 1, 2).to("cuda")
    observed = []

    def network(*parts):
        observed.append((parts[0].stride(), parts[0].shape, parts[0].dtype))
        got = torch.cat(parts, dim=1) if len(parts) == 2 else parts[0]
        np.testing.assert_array_equal(got.permute(0, 2, 3, 1).cpu().numpy(), normalized)
        return output.clone()

    backend.nets[model] = network
    expected = backend.run(model, image.astype(np.float32)).astype(np.uint8)
    actual = backend.run_u8(model, image)
    np.testing.assert_array_equal(actual, expected)
    frozen = actual.copy()
    backend.run_u8(model, image)
    np.testing.assert_array_equal(actual, frozen)
    assert observed[0] == observed[1] == observed[2]
    assert observed[1][0] == (shape[1] * shape[2] * shape[3], 1, shape[2] * shape[3], shape[3])
    assert observed[1][1][0] == 1 and observed[1][2] == torch.float32


@pytest.mark.skipif(not CUDA_OPT_IN, reason="Explicit GPU scheduling required")
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_cuda_bytes_reject_nonfinite_before_quantization(bad):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    backend = TorchBackend("unused", "cuda")
    backend.nets["colorfix"] = lambda pixels: torch.full((1, 3, 544, 544), bad, device="cuda")
    with pytest.raises(RuntimeError, match="output contains NaN or infinity"):
        backend.run_u8("colorfix", np.zeros((1, 544, 544, 6), np.uint8))
