"""Byte-exact export regressions against the frozen 128-row encoder arithmetic."""
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pytest

from m9 import nef
from m9.exif import add_nef_exif
from m9.uhdr import _seg, icc_p3, srgb_to_p3


def _reference(rgb, quality=95):
    pixels = np.empty(rgb.shape, np.uint8)
    for y in range(0, rgb.shape[0], 128):
        pixels[y:y + 128] = (srgb_to_p3(rgb[y:y + 128]) * 255.0 + 0.5).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", pixels[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    data = encoded.tobytes()
    return data[:2] + _seg(0xFFE2, b"ICC_PROFILE\0\x01\x01" + icc_p3()) + data[2:]


@pytest.mark.parametrize("height,width", [(1, 1), (127, 17), (128, 31), (129, 31),
                                         (255, 17), (256, 17), (257, 31)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_random_export_is_full_jpeg_byte_equal_at_strip_boundaries(height, width, dtype):
    rgb = np.random.default_rng(1901).uniform(-.2, 1.2, (height, width, 3)).astype(dtype)
    before = rgb.copy()
    assert nef._encode_jpeg(rgb) == _reference(rgb)
    np.testing.assert_array_equal(rgb, before)


@pytest.mark.parametrize("view", ["reversed_rows", "reversed_columns", "strided", "transposed"])
def test_export_matches_for_noncontiguous_input(view):
    rgb = np.random.default_rng(19).random((259, 263, 3), dtype=np.float32)
    rgb = {"reversed_rows": rgb[::-1], "reversed_columns": rgb[:, ::-1],
           "strided": rgb[::2, ::2], "transposed": rgb.transpose(1, 0, 2)}[view]
    assert not rgb.flags.c_contiguous
    assert nef._encode_jpeg(rgb) == _reference(rgb)


@pytest.mark.parametrize("quality", [90, 95, 100])
def test_transfer_clipping_and_known_rounding_values_keep_encoder_bytes(quality):
    boundaries = np.array([-.5, -0., 0., np.nextafter(np.float32(.04045), np.float32(0)),
                           .04045, np.nextafter(np.float32(.04045), np.float32(1)),
                           .5 / 255, 1.5 / 255, 127.5 / 255, 128.5 / 255,
                           np.nextafter(np.float32(1), np.float32(0)), 1., 1.5], np.float32)
    rgb = np.resize(boundaries, (257, 17, 3))
    assert nef._encode_jpeg(rgb, quality) == _reference(rgb, quality)


def test_quantization_keeps_the_original_separate_float32_multiply_and_add(monkeypatch):
    values = np.array([0, .5 / 255, 1.5 / 255, 127.5 / 255, 128.5 / 255, .5, 1], np.float32)
    rgb = np.repeat(values[None, :, None], 3, axis=2)
    monkeypatch.setattr(nef, "_p3_strip", lambda stripe, *buffers: stripe.copy())
    actual_encode = cv2.imencode
    captured = []

    def encode(extension, pixels, parameters):
        captured.append(pixels.copy())
        return actual_encode(extension, pixels, parameters)

    monkeypatch.setattr(nef.cv2, "imencode", encode)
    nef._encode_jpeg(rgb)
    expected = np.repeat(np.array([0, 1, 2, 128, 129, 128, 255], np.uint8)[None, :, None], 3, axis=2)
    np.testing.assert_array_equal(captured[0], expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("height,width", [(1, 1), (127, 17), (128, 31), (129, 17)])
def test_p3_scratch_float32_bits_match_original_transfer_and_matrix(dtype, height, width):
    rgb = np.random.default_rng(401).uniform(-.2, 1.2, (height * 2, width * 2, 3)).astype(dtype)
    rgb = rgb[::2, ::2]
    before = rgb.copy()
    shape = (min(128, height), width, 3)
    x, work, mask = np.empty(shape, np.float32), np.empty(shape, np.float32), np.empty(shape, bool)
    for y in range(0, height, 128):
        part = rgb[y:y + 128]
        expected = srgb_to_p3(part)
        actual = nef._p3_strip(part, x[:len(part)], work[:len(part)], mask[:len(part)])
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(rgb, before)


def test_p3_scratch_is_bit_exact_at_transfer_thresholds_and_clip_edges():
    values = np.array([-.1, -0., 0., np.nextafter(np.float32(.04045), np.float32(0)), .04045,
                       np.nextafter(np.float32(.04045), np.float32(1)), .0031308, .5, 1., 1.1], np.float32)
    rgb = np.resize(values, (129, 17, 3))
    shape = (128, 17, 3)
    x, work, mask = np.empty(shape, np.float32), np.empty(shape, np.float32), np.empty(shape, bool)
    for y in range(0, 129, 128):
        part = rgb[y:y + 128]
        actual = nef._p3_strip(part, x[:len(part)], work[:len(part)], mask[:len(part)])
        np.testing.assert_array_equal(actual.view(np.uint32), srgb_to_p3(part).view(np.uint32))


def test_export_keeps_pixels_icc_and_exif_independent_of_json(tmp_path):
    rgb = np.random.default_rng(51).random((129, 31, 3), dtype=np.float32)
    baseline, summary_old = add_nef_exif(_reference(rgb), tmp_path / "missing.NEF", (31, 129))
    actual, summary_new = add_nef_exif(nef._encode_jpeg(rgb), tmp_path / "missing.NEF", (31, 129))
    assert actual == baseline and summary_new == summary_old
    with Image.open(BytesIO(actual)) as image:
        assert image.size == (31, 129) and image.info["icc_profile"] == icc_p3()
        assert image.getexif()[274] == 1
        assert image.getexif().get_ifd(34665)[40961] == 65535
    assert not list(tmp_path.iterdir())
