"""Exact frontend arithmetic and strip scheduling, independent of strip height."""
import concurrent.futures
from types import SimpleNamespace

import numpy as np
import pytest

from m9 import frontend
from m9.ae import AEResult


def serial_reference(lin, matrix, info, *, camera="main", lux_index=None, cct=None, ev=0.0, ae=None):
    """The existing IPE arithmetic applied once to the whole image, without a pool."""
    tag = frontend.CAMERAS[camera][0]
    li = float(lux_index) if lux_index is not None else (
        frontend.lux_index_from_ev(info.ev100) if info.ev100 else 300.0)
    k = float(cct) if cct is not None else info.cct
    if ae is None:
        ae = frontend.meter(lin, li, camera)
    gain = np.float32(ae.exp_short * 2.0 ** ev)
    cond = dict(drc=ae.adrc_gain, lux=li, cct=k)
    ccm = frontend.cc_m9(matrix, li, k, ae.adrc_gain, camera).T
    hue, sat = frontend.tuning.tdl_tables(tag, flag=0.0, **cond)
    gamma = frontend.tuning.gamma_lut(tag, **cond)
    cv = frontend.tuning.cv_params(tag, flag=0.0, **cond)
    gtm = frontend.tmc_lut(ae.adrc_gain, li, tag=tag)
    pixels = frontend.ltm_global(lin * gain, ae.adrc_gain, gtm)
    pixels = np.clip(pixels @ ccm, 0, None)
    pixels = frontend.tdl_apply(pixels, hue, sat)
    pixels = frontend.gamma_apply(pixels, gamma)
    pixels = np.clip(frontend.cv_apply(pixels, cv), 0, 1)
    return SimpleNamespace(rgb=pixels, ae=ae, lux_index=li, cct=k)


def assert_result(actual, expected, lin, matrix, info, camera):
    assert actual.rgb.shape == lin.shape and actual.rgb.dtype == np.float32
    np.testing.assert_array_equal(actual.rgb.view(np.uint32), expected.rgb.view(np.uint32))
    assert vars(actual.ae) == vars(expected.ae)
    assert actual.lux_index == expected.lux_index and actual.cct == expected.cct
    assert actual.lin is lin and actual.cam2srgb is matrix and actual.info is info
    assert actual.camera == camera


def fixed_ae():
    return AEResult(300.0, 120.0, 3.0, 26.0, .6, 15.6, .3, 200.0, 180.0, .25, 1.2)


@pytest.mark.parametrize("camera", ["main", "tele"])
@pytest.mark.parametrize("view", ["odd_landscape", "portrait", "reversed", "small_ae"])
def test_real_meter_and_ipe_bits_match_whole_image_serial(monkeypatch, camera, view):
    lin = np.random.default_rng(1801).uniform(.001, 1.1, (67, 99, 3)).astype(np.float32)
    options = {}
    if view == "portrait":
        lin = lin.transpose(1, 0, 2)
        options = dict(lux_index=301.25, cct=4300.5, ev=.3)
    elif view == "reversed":
        lin = lin[::-1, ::-1]
    elif view == "small_ae":
        lin = lin[:17, :19].copy()
    matrix = np.array([[1.1, -.1, .01], [-.05, 1.02, .03], [.01, -.03, 1.04]], np.float32)
    info = frontend.RawInfo(cct=5900.0, ev100=0.0 if view == "reversed" else 12.3,
                            extra={"sentinel": "unchanged"})
    before, matrix_before = lin.copy(), matrix.copy()
    expected = serial_reference(lin, matrix, info, camera=camera, **options)
    original_meter, meter_calls = frontend.meter, []

    def meter(image, lux, selected_camera):
        meter_calls.append((image, lux, selected_camera))
        return original_meter(image, lux, selected_camera)

    monkeypatch.setattr(frontend, "meter", meter)
    monkeypatch.setattr(frontend.os, "cpu_count", lambda: 8)
    actual = frontend.b2y_from_raw(lin, matrix, info, camera=camera, **options)
    assert_result(actual, expected, lin, matrix, info, camera)
    assert len(meter_calls) == 1 and meter_calls[0][0] is lin
    assert meter_calls[0][1:] == (expected.lux_index, camera)
    np.testing.assert_array_equal(lin, before)
    np.testing.assert_array_equal(matrix, matrix_before)
    assert info.extra == {"sentinel": "unchanged"}


def test_short_and_boundary_images_never_submit_empty_ipe_strips(monkeypatch):
    # Fixed AE isolates scheduling from the pre-existing 16x16 metering layout.
    ae = fixed_ae()
    meter_calls, strip_shapes = [], []
    original_tdl = frontend.tdl_apply
    original_executor = concurrent.futures.ThreadPoolExecutor

    class BoundedExecutor(original_executor):
        def map(self, callback, indices, **kwargs):
            indices = tuple(indices)
            assert 1 <= self._max_workers <= len(indices)
            return super().map(callback, indices, **kwargs)

    def meter(image, lux, camera):
        meter_calls.append((image, lux, camera))
        return ae

    def tdl(image, hue, sat):
        assert image.shape[0] > 0 and image.shape[1] > 0
        strip_shapes.append(image.shape[:2])
        return original_tdl(image, hue, sat)

    monkeypatch.setattr(frontend, "meter", meter)
    monkeypatch.setattr(frontend, "tdl_apply", tdl)
    monkeypatch.setattr(frontend.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", BoundedExecutor)
    heights = (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257)
    matrix = np.eye(3, dtype=np.float32)
    info = frontend.RawInfo(cct=5100, ev100=11)
    for index, (height, width) in enumerate([(1, 1)] + [(h, 1 if h == 15 else 17) for h in heights]):
        camera = "main" if index % 2 else "tele"
        lin = np.random.default_rng(height + width).uniform(0, 1.2, (height, width, 3)).astype(np.float32)
        before = lin.copy()
        expected = serial_reference(lin, matrix, info, camera=camera, ae=ae)
        meter_calls.clear()
        strip_shapes.clear()
        actual = frontend.b2y_from_raw(lin, matrix, info, camera=camera)
        assert_result(actual, expected, lin, matrix, info, camera)
        assert actual.ae is ae
        assert len(meter_calls) == 1 and meter_calls[0][0] is lin
        assert sum(shape[0] for shape in strip_shapes) == height
        assert all(shape[1] == width for shape in strip_shapes)
        np.testing.assert_array_equal(lin, before)


def identity_pixel_chain(monkeypatch, visited):
    """Distinct row values make omissions/overlaps visible without real AE limits."""
    ae = fixed_ae()
    ae.exp_short = ae.adrc_gain = 1.0
    monkeypatch.setattr(frontend, "meter", lambda *args: ae)
    monkeypatch.setattr(frontend, "cc_m9", lambda *args: np.eye(3, dtype=np.float32))
    monkeypatch.setattr(frontend.tuning, "tdl_tables", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(frontend.tuning, "gamma_lut", lambda *args, **kwargs: None)
    monkeypatch.setattr(frontend.tuning, "cv_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(frontend, "tmc_lut", lambda *args, **kwargs: None)
    monkeypatch.setattr(frontend, "ltm_global", lambda pixels, *args: pixels)
    monkeypatch.setattr(frontend, "gamma_apply", lambda pixels, *args: pixels)
    monkeypatch.setattr(frontend, "cv_apply", lambda pixels, *args: pixels)

    def tdl(pixels, *args):
        assert pixels.shape[0] > 0
        visited.extend(pixels[:, 0, 0].tolist())
        return pixels

    monkeypatch.setattr(frontend, "tdl_apply", tdl)


def test_reverse_task_completion_preserves_each_row_once(monkeypatch):
    visited, submissions = [], []
    identity_pixel_chain(monkeypatch, visited)
    monkeypatch.setattr(frontend.os, "cpu_count", lambda: 8)

    class ReverseExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, callback, indices):
            indices = list(indices)
            submissions.append(indices)
            results = {index: callback(index) for index in reversed(indices)}
            return iter(results[index] for index in indices)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", ReverseExecutor)
    rows = np.arange(1, 1026, dtype=np.float32) / np.float32(2048)
    lin = np.broadcast_to(rows[:, None, None], (len(rows), 17, 3)).copy()
    result = frontend.b2y_from_raw(lin, np.eye(3, dtype=np.float32), frontend.RawInfo(), camera="tele")
    np.testing.assert_array_equal(result.rgb.view(np.uint32), lin.view(np.uint32))
    assert sorted(visited) == rows.tolist() and len(visited) == len(rows)
    assert any(len(indices) > 1 for indices in submissions)
    assert visited != rows.tolist()


def test_strip_error_propagates_after_executor_shutdown(monkeypatch):
    visited, executors = [], []
    identity_pixel_chain(monkeypatch, visited)
    monkeypatch.setattr(frontend.os, "cpu_count", lambda: 8)
    original_executor = concurrent.futures.ThreadPoolExecutor

    class RecordingExecutor(original_executor):
        def __enter__(self):
            executors.append(self)
            return super().__enter__()

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", RecordingExecutor)
    failure = RuntimeError("deliberate IPE strip failure")

    def fail(pixels, *args):
        raise failure

    monkeypatch.setattr(frontend, "tdl_apply", fail)
    lin = np.random.default_rng(29).random((1025, 17, 3), dtype=np.float32)
    before = lin.copy()
    with pytest.raises(RuntimeError, match="deliberate IPE strip failure") as caught:
        frontend.b2y_from_raw(lin, np.eye(3, dtype=np.float32), frontend.RawInfo())
    assert caught.value is failure
    assert all(not worker.is_alive() for executor in executors for worker in executor._threads)
    np.testing.assert_array_equal(lin, before)
