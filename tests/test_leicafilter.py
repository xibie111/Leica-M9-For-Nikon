import numpy as np
import pytest

from m9 import leicafilter


@pytest.mark.parametrize("shape", [(73, 119), (119, 73), (7, 1)])
@pytest.mark.parametrize("current_time", [0., 12.3])
def test_chunked_vignette_matches_whole_image_coordinates(shape, current_time):
    torch = pytest.importorskip("torch")
    rgb = np.random.default_rng(9).random((*shape, 3), dtype=np.float32)
    # Nonzero coordinate noise makes resetting the row origin fail this comparison.
    parameters = np.array([.1, .7, 1.2, .65, 7., .3, .7, .1], np.float32)
    full = leicafilter._cvstyle_torch(torch, "cpu", rgb, parameters, current_time,
                                    chunk_rows=shape[0])
    chunked = leicafilter._cvstyle_torch(torch, "cpu", rgb, parameters, current_time,
                                       chunk_rows=7)
    np.testing.assert_allclose(chunked, full, rtol=0, atol=1e-7)


def test_vignette_row_allocation_is_bounded(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    def rows(torch, dev, block, p, current_time, y0, h):
        calls.append((block.shape, y0, h))
        return block

    monkeypatch.setattr(leicafilter, "_cvstyle_torch_rows", rows)
    # Width and height yield a non-full final block without large memory use.
    image = np.zeros((1050, 1000, 3), np.float32)
    result = leicafilter._cvstyle_torch(torch, "cpu", image, np.zeros(8), 0.)
    np.testing.assert_array_equal(result, image)
    assert calls == [((1048, 1000, 3), 0, 1050), ((2, 1000, 3), 1048, 1050)]


def test_explicit_filter_device_reaches_both_stages(monkeypatch):
    seen = []
    monkeypatch.setattr(leicafilter, "lut_apply",
                        lambda rgb, lut, device=None: seen.append(("lut", device)) or rgb)
    monkeypatch.setattr(leicafilter, "cvstyle_apply",
                        lambda rgb, parameters, device=None: seen.append(("vignette", device)) or rgb)
    image = np.zeros((9, 11, 3), np.float32)
    result = leicafilter.LeicaFilter(device="cpu")(image, 200, 5000)
    np.testing.assert_array_equal(result, image)
    assert seen == [("lut", "cpu"), ("vignette", "cpu")]


def test_torch_lut_matches_numpy_path(monkeypatch):
    pytest.importorskip("torch")
    rgb = np.random.default_rng(5).random((17, 23, 3), dtype=np.float32)
    lut = np.random.default_rng(6).integers(0, 256, (17, 17, 17, 3), dtype=np.uint8)
    accelerated = leicafilter.lut_apply(rgb, lut, device="cpu")
    monkeypatch.setattr(leicafilter, "_torch", lambda device=None: (None, None))
    reference = leicafilter.lut_apply(rgb, lut, device="cpu")
    np.testing.assert_allclose(accelerated, reference, rtol=0, atol=2e-7)
