from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from m9 import frontend, raw_input
from m9.nikon_matrices import nikon_matrix_catalog


# LibRaw Nikon Z 7 / Z 7_2 table. Its asymmetry detects an accidental transpose.
NIKON_XYZ_TO_CAM = np.array([[1.3705, -.6004, -.1400],
                            [-.5464, 1.3568, .2062],
                            [-.0940, .1706, .7618]])
Z6_3_XYZ_TO_CAM = np.array([[11206, -4286, -941], [-4879, 12847, 2251],
                           [-745, 1654, 7374]], dtype=np.float32) / np.float32(10000)


def tags():
    return dict(Make="NIKON CORPORATION", Model="NIKON Z 7_2", ExposureTime=(1, 125),
                FNumber=(28, 10), ISOSpeedRatings=200, Orientation=1, DNGVersion=b"\x01\x04\0\0",
                ColorMatrix1=NIKON_XYZ_TO_CAM, AsShotNeutral=[.5, 1., .7],
                PhotometricInterpretation=32803, CFAPattern=b"\0\1\1\2", SamplesPerPixel=1,
                Compression=7, ImageWidth=8, ImageLength=6)


class FakeRaw:
    num_colors = 3
    color_desc = b"RGBG"
    raw_pattern = np.array([[0, 1], [3, 2]])
    rgb_xyz_matrix = np.vstack([NIKON_XYZ_TO_CAM, np.zeros(3)])

    def __init__(self, flip=0):
        self.sizes = SimpleNamespace(raw_width=10, raw_height=8, width=8, height=6,
                                     left_margin=1, top_margin=1, iwidth=8, iheight=6, flip=flip)
        neutral = NIKON_XYZ_TO_CAM @ np.array([.95047, 1., 1.08883])
        self.camera_whitebalance = [neutral[1] / x for x in neutral] + [1.]
        self.kwargs = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def postprocess(self, **kwargs):
        self.kwargs = kwargs
        shape = (8, 6, 3) if self.sizes.flip & 4 else (6, 8, 3)
        return np.full(shape, 32768, dtype=np.uint16)


@pytest.fixture
def nef(tmp_path, monkeypatch):
    path = tmp_path / "中文 目录" / "DSC 0001.NEF"
    path.parent.mkdir()
    path.write_bytes(b"test NEF")
    monkeypatch.setattr(frontend, "dng_tags", lambda _: tags())
    return path


@pytest.fixture
def z6_zero_case(nef, monkeypatch):
    metadata = tags()
    metadata.update(Model="NIKON Z6_3", Compression=34713, StripOffsets=(0,))
    raw = FakeRaw()
    raw.rgb_xyz_matrix = np.zeros((4, 3), np.float32)
    neutral = Z6_3_XYZ_TO_CAM.astype(np.float64) @ np.array([.95047, 1., 1.08883])
    raw.camera_whitebalance = [neutral[1] / x for x in neutral] + [1.]
    monkeypatch.setattr(frontend, "dng_tags", lambda _: metadata)
    monkeypatch.setattr(raw_input.rawpy, "imread", lambda _: raw)
    converter = Mock(side_effect=AssertionError("native processing must never invoke the converter"))
    monkeypatch.setattr(raw_input, "convert_nef_to_dng", converter)
    yield nef, raw, metadata
    converter.assert_not_called()


@pytest.mark.parametrize("model", ["NIKON Z6_3", "  nikon z6_3  ", "NIKON Z 6 III"])
def test_native_z6_zero_matrix_uses_only_exact_upstream_profile(z6_zero_case, model):
    path, raw, metadata = z6_zero_case
    metadata["Model"] = model
    progress = Mock()
    rgb, matrix, info = raw_input.load_nef(path, method="native", progress=progress)
    expected_wb, expected_matrix = frontend.camera_d50_transform(Z6_3_XYZ_TO_CAM)
    assert rgb.dtype == np.float32 and rgb.shape == (6, 8, 3)
    np.testing.assert_allclose(matrix, expected_matrix, rtol=0, atol=1e-7)
    np.testing.assert_allclose(raw.kwargs["user_wb"][:3], expected_wb, rtol=0, atol=1e-7)
    assert raw.kwargs["output_color"] == raw_input.rawpy.ColorSpace.raw
    assert raw.kwargs["gamma"] == (1, 1) and raw.kwargs["half_size"] is False
    assert info.cct == pytest.approx(6503, abs=30)
    assert info.exposure_time == 1 / 125 and info.iso == 200 and info.fnumber == 2.8
    assert info.extra["method"] == "native"
    assert "0.22.1" in info.extra["matrix_source"] and "XYZ -> camera" in info.extra["matrix_source"]
    profile = info.extra["matrix_compatibility_profile"]
    np.testing.assert_array_equal(np.asarray(profile["xyz_to_camera"], np.float32), Z6_3_XYZ_TO_CAM)
    assert profile["camera_model"] == "NIKON Z6_3"
    assert profile["upstream_libraw_version"] == "0.22.1"
    assert profile["matched_model"] == model.strip()
    assert profile["reason"] == "installed LibRaw returned an all-zero cam_xyz"
    assert profile["source_url"] == "https://github.com/LibRaw/LibRaw/blob/0.22.1/src/tables/colordata.cpp#L1036"
    assert profile["raw_strip_signature"] == b"test".hex()
    assert any("0.22.1" in message and "Z6_3" in message for _, message in
               (call.args for call in progress.call_args_list))


@pytest.mark.parametrize("model", ["NIKON Z6_3", "NIKON Z 7_2", "NIKON COOLPIX 700", "NIKON UNKNOWN"])
def test_native_valid_existing_matrix_is_preserved_without_compatibility_profile(z6_zero_case, monkeypatch, model):
    path, raw, metadata = z6_zero_case
    metadata["Model"] = model
    metadata.pop("StripOffsets")  # Valid installed matrices do not need this compatibility guard.
    raw.rgb_xyz_matrix = np.vstack([NIKON_XYZ_TO_CAM, np.zeros(3)])
    raw.camera_whitebalance = FakeRaw().camera_whitebalance
    lookup = Mock(side_effect=AssertionError("valid installed matrices must not consult the backport"))
    monkeypatch.setattr(raw_input, "lookup_nikon_matrix", lookup)
    _, matrix, info = raw_input.load_nef(path, method="native")
    expected_wb, expected_matrix = frontend.camera_d50_transform(NIKON_XYZ_TO_CAM)
    np.testing.assert_allclose(matrix, expected_matrix, rtol=0, atol=1e-7)
    np.testing.assert_allclose(raw.kwargs["user_wb"][:3], expected_wb, rtol=0, atol=1e-7)
    assert "matrix_compatibility_profile" not in info.extra
    assert info.extra["matrix_source"] == "LibRaw cam_xyz (XYZ -> camera), single matrix"
    lookup.assert_not_called()


@pytest.mark.parametrize("model", ["NIKON UNKNOWN", "NIKON D50X", "NIKON D5000garbage",
                                  "NIKON Z6_3 stale", "NIKON Z 6_3"])
def test_native_unknown_or_wrong_model_zero_matrix_is_not_repaired(z6_zero_case, model):
    path, raw, metadata = z6_zero_case
    metadata["Model"] = model
    with pytest.raises(ValueError, match="matrix"):
        raw_input.load_nef(path, method="native")
    assert raw.kwargs is None


@pytest.mark.parametrize("make", ["NIKON CAMERAS", "NIKON CORPORATION LTD", "NOT NIKON"])
def test_native_zero_matrix_does_not_accept_a_make_containing_nikon(z6_zero_case, make):
    path, raw, metadata = z6_zero_case
    metadata["Make"] = make
    with pytest.raises(ValueError, match="matrix"):
        raw_input.load_nef(path, method="native")
    assert raw.kwargs is None


@pytest.mark.parametrize("entry", [entry for entry in nikon_matrix_catalog()["entries"]
                                 if len(entry["coefficients"]) == 9], ids=lambda entry: entry["model_prefix"])
def test_native_all_known_three_channel_profiles_handle_missing_matrix(z6_zero_case, entry):
    path, raw, metadata = z6_zero_case
    metadata["Model"] = "NIKON " + entry["models"][0]
    expected = np.asarray(entry["coefficients"], np.float32).reshape(3, 3) / np.float32(10000)
    neutral = expected.astype(np.float64) @ np.array([.95047, 1., 1.08883])
    raw.camera_whitebalance = [neutral[1] / value for value in neutral] + [1.]
    rgb, matrix, info = raw_input.load_nef(path, method="native")
    expected_wb, expected_matrix = frontend.camera_d50_transform(expected)
    np.testing.assert_array_equal(matrix, expected_matrix)
    np.testing.assert_array_equal(np.asarray(info.extra["matrix_compatibility_profile"]["xyz_to_camera"], np.float32), expected)
    np.testing.assert_allclose(raw.kwargs["user_wb"][:3], expected_wb, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(rgb, np.full((6, 8, 3), np.float32(32768) / np.float32(65535)))
    assert info.cct == pytest.approx(6503, abs=30)
    assert info.extra["matrix_compatibility_profile"]["camera_model"] == "NIKON " + entry["model_prefix"]


@pytest.mark.parametrize("entry", [entry for entry in nikon_matrix_catalog()["entries"]
                                 if len(entry["coefficients"]) == 12], ids=lambda entry: entry["model_prefix"])
def test_native_four_channel_profile_is_never_truncated_to_three(z6_zero_case, entry):
    path, raw, metadata = z6_zero_case
    metadata["Model"] = "NIKON " + entry["models"][0]
    with pytest.raises(ValueError, match="four colour channels"):
        raw_input.load_nef(path, method="native")
    assert raw.kwargs is None


@pytest.mark.parametrize("failure", ["nan", "inf", "singular", "tiny_nonzero", "bad_last_row", "shape"])
def test_native_z6_nonzero_or_malformed_matrix_is_not_replaced(z6_zero_case, failure):
    path, raw, _ = z6_zero_case
    matrix = np.zeros((4, 3), np.float64)
    if failure == "nan":
        matrix[0, 0] = np.nan
    elif failure == "inf":
        matrix[0, 0] = np.inf
    elif failure == "singular":
        matrix[:3] = 1
    elif failure == "tiny_nonzero":
        matrix[0, 0] = 1e-30
    elif failure == "bad_last_row":
        matrix[3, 0] = 1
    else:
        matrix = np.zeros((3, 3))
    raw.rgb_xyz_matrix = matrix
    with pytest.raises(ValueError, match="matrix"):
        raw_input.load_nef(path, method="native")
    assert raw.kwargs is None


@pytest.mark.parametrize("key,value", [
    ("Compression", 7), ("Compression", 34713.5), ("Compression", 1.5),
    ("PhotometricInterpretation", 2), ("PhotometricInterpretation", 32803.5),
    ("SamplesPerPixel", 3), ("SamplesPerPixel", 1.5),
    ("StripOffsets", ()), ("StripOffsets", (0, 4)), ("StripOffsets", (-1,)),
    ("StripOffsets", (.5,)), ("StripOffsets", (np.inf,)), ("StripOffsets", (np.nan,)),
    ("StripOffsets", None), ("StripOffsets", (1000,)), ("StripOffsets", ("invalid",)),
])
def test_native_z6_profile_rejects_nonstandard_or_malformed_raw_header(z6_zero_case, key, value):
    path, raw, metadata = z6_zero_case
    metadata[key] = value
    with pytest.raises(ValueError):
        raw_input.load_nef(path, method="native")
    assert raw.kwargs is None


def test_native_uncompressed_profile_does_not_mistake_pixels_for_he_magic(z6_zero_case):
    path, raw, metadata = z6_zero_case
    metadata["Compression"] = 1
    path.write_bytes(b"\xff\x10\xff\x50pixels")
    _, _, info = raw_input.load_nef(path, method="native")
    assert raw.kwargs is not None
    assert info.extra["matrix_compatibility_profile"]["raw_strip_signature"] == "ff10ff50"


def test_native_uncompressed_profile_also_rejects_incomplete_strip(z6_zero_case):
    path, raw, metadata = z6_zero_case
    metadata["Compression"] = 1
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="incomplete"):
        raw_input.load_nef(path, method="native")
    assert raw.kwargs is None


@pytest.mark.parametrize("data,offset", [(b"abc", 0), (b"header" + b"\xff\x10\xff\x50" + b"tail", 6)])
def test_native_z6_profile_rejects_short_strip_or_jpeg_xs_he(z6_zero_case, data, offset):
    path, raw, metadata = z6_zero_case
    path.write_bytes(data)  # Generated fixture only; no user RAW is modified.
    metadata["StripOffsets"] = (offset,)
    with pytest.raises(ValueError, match="HE|incomplete"):
        raw_input.load_nef(path, method="native")
    assert raw.kwargs is None


def test_native_z6_profile_checks_actual_raw_subifd_and_strip_offset(z6_zero_case):
    path, _, metadata = z6_zero_case
    raw_tags = {k: metadata[k] for k in ("Compression", "PhotometricInterpretation", "SamplesPerPixel", "StripOffsets")}
    raw_tags["StripOffsets"] = (6,)
    metadata.update(Compression=1, PhotometricInterpretation=2, SamplesPerPixel=3, _raw_ifd=raw_tags)
    path.write_bytes(b"header" + b"\x01\x02\x03\x04" + b"tail")
    _, _, info = raw_input.load_nef(path, method="native")
    assert info.extra["matrix_compatibility_profile"]["raw_strip_signature"] == "01020304"


@pytest.mark.parametrize("failure", ["zero_wb", "nan_wb", "nonphysical_cct", "not_bayer",
                                     "four_colors", "not_rgb", "wrong_cfa_shape"])
def test_native_z6_profile_retains_white_balance_scene_and_cfa_guards(z6_zero_case, failure):
    path, raw, _ = z6_zero_case
    if failure == "zero_wb":
        raw.camera_whitebalance = [0, 1, 1, 1]
    elif failure == "nan_wb":
        raw.camera_whitebalance = [np.nan, 1, 1, 1]
    elif failure == "nonphysical_cct":
        raw.camera_whitebalance = [10, 1, 10, 1]
    elif failure == "not_bayer":
        raw.raw_pattern = None
    elif failure == "four_colors":
        raw.num_colors = 4
    elif failure == "not_rgb":
        raw.color_desc = b"CMYG"
    else:
        raw.raw_pattern = np.zeros((3, 3))
    with pytest.raises(ValueError):
        raw_input.load_nef(path, method="native")
    assert raw.kwargs is None


@pytest.mark.parametrize("flip,width,height,crop", [(0, 8, 6, (2, 2, 5, 3)), (5, 6, 4, (2, 2, 3, 2))])
def test_native_z6_profile_preserves_camera_crop_and_orientation(z6_zero_case, flip, width, height, crop):
    path, raw, _ = z6_zero_case
    raw.sizes = SimpleNamespace(raw_width=width + 2, raw_height=height + 2, width=width, height=height,
                                left_margin=1, top_margin=1, iwidth=width, iheight=height, flip=flip,
                                crop_left_margin=crop[0], crop_top_margin=crop[1], crop_width=crop[2], crop_height=crop[3])
    sensor = np.arange(height * width * 3, dtype=np.uint16).reshape(height, width, 3)

    def render(**kwargs):
        raw.kwargs = kwargs
        return raw_input._orient_pixels(sensor, flip)

    raw.postprocess = render
    rgb, _, info = raw_input.load_nef(path, method="native")
    expected = raw_input._orient_pixels(sensor[1:1 + crop[3], 1:1 + crop[2]], flip).astype(np.float32) / 65535
    np.testing.assert_array_equal(rgb, expected)
    assert info.extra["source_camera_rectangle"] == list(crop)


def test_native_uses_libraw_matrix_in_actual_direction_and_d50(nef, monkeypatch):
    raw = FakeRaw()
    monkeypatch.setattr(raw_input.rawpy, "imread", lambda _: raw)
    progress = Mock()
    rgb, cam2srgb, info = raw_input.load_nef(nef, method="native", progress=progress)
    assert rgb.dtype == np.float32 and rgb.shape == (6, 8, 3)
    expected_mul, expected_matrix = frontend.camera_d50_transform(NIKON_XYZ_TO_CAM)
    np.testing.assert_allclose(cam2srgb, expected_matrix, atol=1e-6)
    np.testing.assert_allclose(raw.kwargs["user_wb"][:3], expected_mul, atol=1e-6)
    assert raw.kwargs["output_color"] == raw_input.rawpy.ColorSpace.raw
    assert raw.kwargs["gamma"] == (1, 1) and raw.kwargs["half_size"] is False
    assert info.cct == pytest.approx(6503, abs=30)
    assert info.ev100 == pytest.approx(np.log2(2.8 ** 2 * 125) - 1)
    assert info.extra["cct_is_approximate"] is True
    assert "XYZ -> camera" in info.extra["matrix_source"]
    assert progress.call_args_list[0].args[0] == "metadata"


def test_native_rotated_dimensions_follow_libraw(nef, monkeypatch):
    monkeypatch.setattr(raw_input.rawpy, "imread", lambda _: FakeRaw(flip=6))
    rgb, _, info = raw_input.load_nef(nef, method="native")
    assert rgb.shape == (8, 6, 3)
    assert info.extra["decoded_dimensions"] == [6, 8]


@pytest.mark.parametrize("flip", range(8))
def test_native_preserves_camera_inset_photo_and_orientation(nef, monkeypatch, flip):
    raw = FakeRaw(flip=flip)
    raw.sizes.crop_left_margin, raw.sizes.crop_top_margin = 2, 2
    raw.sizes.crop_width, raw.sizes.crop_height = 5, 3
    sensor = np.arange(6 * 8 * 3, dtype=np.uint16).reshape(6, 8, 3)
    raw.postprocess = lambda **kw: raw_input._orient_pixels(sensor, flip)
    monkeypatch.setattr(raw_input.rawpy, "imread", lambda _: raw)
    rgb, _, info = raw_input.load_nef(nef, method="native")
    expected = raw_input._orient_pixels(sensor[1:4, 1:6], flip).astype(np.float32) / 65535.
    np.testing.assert_array_equal(rgb, expected)
    assert info.extra["source_camera_rectangle"] == [2, 2, 5, 3]
    assert info.extra["raw_width"] == 10 and info.extra["width"] == 8


@pytest.mark.parametrize("crop", [(2, 2, 0, 3), (0, 0, 5, 3), (2, 2, 10, 10)])
def test_invalid_camera_inset_crop_is_explicit(crop):
    geometry = dict(raw_width=10, raw_height=8, width=8, height=6,
                    left_margin=1, top_margin=1, iwidth=8, iheight=6, flip=0)
    geometry.update(zip(("crop_left_margin", "crop_top_margin", "crop_width", "crop_height"), crop))
    with pytest.raises(ValueError, match="camera crop"):
        raw_input._photo_geometry(geometry)


def test_dng_route_targets_source_camera_crop(nef, monkeypatch):
    raw = FakeRaw()
    raw.sizes.crop_left_margin, raw.sizes.crop_top_margin = 2, 2
    raw.sizes.crop_width, raw.sizes.crop_height = 5, 3
    monkeypatch.setattr(raw_input.rawpy, "imread", lambda _: raw)
    metadata = tags()
    metadata.update(DefaultCropOrigin=(1, 1, 1, 1), DefaultCropSize=(5, 1, 3, 1))
    monkeypatch.setattr(frontend, "dng_tags", lambda _: metadata)

    @contextmanager
    def convert(*args, **kwargs):
        yield nef.with_suffix(".dng")

    info = raw_input._exposure_info(metadata)
    info.extra.update(frontend.raw_geometry(raw))
    pixels = np.arange(6 * 8 * 3, dtype=np.float32).reshape(6, 8, 3)
    monkeypatch.setattr(raw_input, "convert_nef_to_dng", convert)
    monkeypatch.setattr(raw_input, "find_dng_converter", lambda _: Path("converter.exe"))
    monkeypatch.setattr(frontend, "load_dng", lambda _: (pixels, np.eye(3), info))
    rgb, _, result = raw_input.load_nef(nef, method="dng")
    np.testing.assert_array_equal(rgb, pixels[1:4, 1:6])
    assert result.extra["source_active_dimensions"] == [5, 3]
    assert result.extra["source_camera_rectangle"] == [2, 2, 5, 3]


@pytest.mark.parametrize("failure", ["matrix", "wb", "bayer", "dimensions"])
def test_native_rejects_invalid_data_without_fallback(nef, monkeypatch, failure):
    raw = FakeRaw()
    if failure == "matrix":
        raw.rgb_xyz_matrix = np.zeros((4, 3))
    elif failure == "wb":
        raw.camera_whitebalance = [0., 1., 1., 1.]
    elif failure == "bayer":
        raw.raw_pattern = None
    else:
        raw.postprocess = lambda **kw: np.zeros((4, 5, 3), np.uint16)
    monkeypatch.setattr(raw_input.rawpy, "imread", lambda _: raw)
    fallback = Mock(side_effect=AssertionError("must not convert"))
    monkeypatch.setattr(raw_input, "convert_nef_to_dng", fallback)
    with pytest.raises(ValueError):
        raw_input.load_nef(nef, method="native")
    fallback.assert_not_called()


@pytest.mark.parametrize("key,value", [("ExposureTime", (0, 1)), ("FNumber", float("nan")),
                                       ("ISOSpeedRatings", 0), ("Orientation", 9)])
def test_invalid_metadata_stops_before_decode(nef, monkeypatch, key, value):
    metadata = tags()
    metadata[key] = value
    monkeypatch.setattr(frontend, "dng_tags", lambda _: metadata)
    decode = Mock()
    monkeypatch.setattr(raw_input.rawpy, "imread", decode)
    with pytest.raises(ValueError, match="metadata"):
        raw_input.load_nef(nef, method="native")
    decode.assert_not_called()


@pytest.mark.parametrize("orientation", [0, 9, "garbage", "6", 6.9, None, True])
def test_malformed_orientation_is_not_defaulted(orientation):
    metadata = tags()
    metadata["Orientation"] = orientation
    with pytest.raises(ValueError, match="Orientation"):
        raw_input._exposure_info(metadata)


def test_dng_dual_matrix_rejects_nonphysical_scene_white():
    matrix = np.array([[2., 1., 0.], [3., 1., 0.], [4., 1., 1.]])
    metadata = {"ColorMatrix1": matrix, "ColorMatrix2": matrix,
                "CalibrationIlluminant1": 21, "CalibrationIlluminant2": 17}
    frontend.camera_d50_transform(matrix)  # valid D50 transform does not imply a valid scene neutral
    with pytest.raises(ValueError, match="scene white"):
        frontend.dng_color(metadata, np.array([1., 2., 3.]))


@pytest.mark.parametrize("flip", range(8))
def test_dng_exact_source_crop_mapping_preserves_pixels(flip):
    # Distinct per-pixel values detect row/column swaps, wrong crop offsets and resampling.
    sensor = np.arange(8 * 10 * 3, dtype=np.float32).reshape(8, 10, 3)
    geometry = dict(raw_width=10, raw_height=8, left_margin=0, top_margin=0,
                    width=10, height=8, flip=flip)
    source = dict(geometry, left_margin=2, top_margin=1, width=6, height=5)
    info = frontend.RawInfo(extra=dict(geometry))
    converted = raw_input._orient_pixels(sensor, flip)
    metadata = {"DefaultCropOrigin": (2, 1), "DefaultCropSize": (6, 5)}
    result = raw_input._reconcile_dng_dimensions(converted, source, info, metadata)
    expected = sensor[1:6, 2:8]
    if flip & 1:
        expected = expected[:, ::-1]
    if flip & 2:
        expected = expected[::-1]
    if flip & 4:
        expected = expected.transpose(1, 0, 2)
    np.testing.assert_array_equal(result, expected)
    assert info.extra["dng_crop_reconciliation"]["source_sensor_rectangle"] == [2, 1, 6, 5]
    assert result.flags.owndata


@pytest.mark.parametrize("failure", ["missing_crop", "wrong_crop", "cropped_pixels", "different_sensor", "fractional"])
def test_dng_ambiguous_or_lost_source_pixels_are_rejected(failure):
    geometry = dict(raw_width=10, raw_height=8, left_margin=0, top_margin=0,
                    width=10, height=8, flip=0)
    source = dict(geometry, left_margin=2, top_margin=1, width=6, height=5)
    metadata = {"DefaultCropOrigin": (2, 1), "DefaultCropSize": (6, 5)}
    if failure == "missing_crop":
        metadata.pop("DefaultCropSize")
    elif failure == "wrong_crop":
        metadata["DefaultCropSize"] = (5, 4)
    elif failure == "cropped_pixels":
        geometry["width"] = 7
    elif failure == "different_sensor":
        geometry["raw_width"] = 12
    else:
        metadata["DefaultCropOrigin"] = (2.5, 1)
    rgb = np.zeros((geometry["height"], geometry["width"], 3), np.float32)
    with pytest.raises(ValueError):
        raw_input._reconcile_dng_dimensions(rgb, source, frontend.RawInfo(extra=geometry), metadata)


def test_damaged_nef_is_explicit_error(nef, monkeypatch):
    monkeypatch.setattr(frontend, "dng_tags", Mock(side_effect=tifffile.TiffFileError("bad TIFF header")))
    with pytest.raises(ValueError, match="Cannot read NEF metadata"):
        raw_input.load_nef(nef)


def test_dng_route_uses_existing_frontend_and_preserves_source_size(nef, monkeypatch):
    monkeypatch.setattr(raw_input.rawpy, "imread", lambda _: FakeRaw())
    converted = nef.with_suffix(".dng")
    converted.write_bytes(b"DNG")

    @contextmanager
    def convert(*args, **kwargs):
        yield converted

    info = raw_input._exposure_info(tags())
    linear = np.ones((6, 8, 3), np.float32)
    decode = Mock(return_value=(linear, np.eye(3, dtype=np.float32), info))
    monkeypatch.setattr(raw_input, "convert_nef_to_dng", convert)
    monkeypatch.setattr(frontend, "load_dng", decode)
    monkeypatch.setattr(raw_input, "find_dng_converter", lambda _: Path("converter.exe"))
    rgb, _, result = raw_input.load_nef(nef, method="dng")
    assert rgb is linear
    decode.assert_called_once_with(str(converted))
    assert result.extra["source_active_dimensions"] == [8, 6]
    decode.return_value = (np.ones((3, 4, 3)), np.eye(3), info)
    with pytest.raises(ValueError, match="dimensions"):
        raw_input.load_nef(nef)


@pytest.mark.parametrize("key,value,message", [
    ("Compression", 34892, "lossless"), ("PhotometricInterpretation", 34892, "CFA"),
    ("ColorMatrix1", np.zeros((3, 3)), "singular"), ("AsShotNeutral", [0, 1, 1], "AsShotNeutral"),
    ("ImageWidth", 0, "dimensions"), ("ExposureTime", (1, 60), "exposure"),
    ("Orientation", 6, "orientation")])
def test_converter_output_validation_rejects_bad_raw(key, value, message, monkeypatch):
    metadata = tags()
    source = raw_input._exposure_info(metadata)
    metadata[key] = value
    monkeypatch.setattr(frontend, "dng_tags", lambda _: metadata)
    with pytest.raises(ValueError, match=message):
        raw_input.validate_converted_dng("converted.dng", source)


def _exposure_mosaic_pair(monkeypatch):
    source = FakeRaw()
    converted = FakeRaw()
    source.raw_image = np.arange(80, dtype=np.uint16).reshape(8, 10)
    converted.raw_image = source.raw_image.copy()
    source.black_level_per_channel = converted.black_level_per_channel = [10] * 4
    monkeypatch.setattr(raw_input.rawpy, "imread", lambda p: source if str(p).lower().endswith(".nef") else converted)
    metadata = tags()
    metadata["ExposureTime"] = (8, 10)
    metadata["ShutterSpeedValue"] = (321928, 1000000)
    source_tags = dict(metadata, ExposureTime=(10, 13))
    source_tags.pop("ShutterSpeedValue")
    return source, converted, source_tags, metadata


def test_shutter_metadata_change_requires_identical_mosaic_and_preserves_source(nef, monkeypatch):
    source, converted, source_tags, metadata = _exposure_mosaic_pair(monkeypatch)
    monkeypatch.setattr(frontend, "dng_tags", lambda p: source_tags if str(p).lower().endswith(".nef") else metadata)

    @contextmanager
    def convert(*args, **kwargs):
        yield nef.with_suffix(".dng")

    info = raw_input._exposure_info(metadata)
    pixels = np.ones((6, 8, 3), np.float32)
    monkeypatch.setattr(raw_input, "convert_nef_to_dng", convert)
    monkeypatch.setattr(raw_input, "find_dng_converter", lambda _: Path("converter.exe"))
    monkeypatch.setattr(frontend, "load_dng", lambda _: (pixels, np.eye(3), info))
    _, _, result = raw_input.load_nef(nef, method="dng")
    assert result.exposure_time == 10 / 13
    assert result.ev100 == raw_input._exposure_info(source_tags).ev100
    assert result.extra["converted_exposure_seconds"] == .8
    assert result.extra["exposure_time_rational"] == [10, 13]
    proof = result.extra["exposure_metadata_reconciliation"]
    assert proof["mosaic_sha256"] == __import__("hashlib").sha256(source.raw_image.data).hexdigest()
    assert proof["source_exposure_time_rational"] == [10, 13]
    assert proof["converted_exposure_time_rational"] == [8, 10]


@pytest.mark.parametrize("failure", ["pixel", "shape", "dtype", "black", "pattern", "apex", "missing_apex", "iso"])
def test_shutter_reconciliation_rejects_unproven_changes(monkeypatch, failure):
    source, converted, source_tags, metadata = _exposure_mosaic_pair(monkeypatch)
    if failure == "pixel":
        converted.raw_image[0, 0] += 1
    elif failure == "shape":
        converted.raw_image = converted.raw_image[:-1]
    elif failure == "dtype":
        converted.raw_image = converted.raw_image.astype(np.uint8)
    elif failure == "black":
        converted.black_level_per_channel = [11] * 4
    elif failure == "pattern":
        converted.raw_pattern = np.array([[2, 1], [3, 0]])
    elif failure == "apex":
        # Its reducible fraction must not enlarge the stored one-millionth APEX quantum.
        metadata["ShutterSpeedValue"] = (321920, 1000000)
    elif failure == "missing_apex":
        metadata.pop("ShutterSpeedValue")
    else:
        metadata["ISOSpeedRatings"] = 250
    monkeypatch.setattr(frontend, "dng_tags", lambda _: metadata)
    with pytest.raises(ValueError):
        raw_input.validate_converted_dng("converted.dng", raw_input._exposure_info(source_tags), source_path="source.nef")


def test_converter_invocation_unicode_path_and_cleanup(nef, monkeypatch, tmp_path):
    executable = tmp_path / "Adobe DNG Converter.exe"
    executable.write_bytes(b"fake converter")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        (Path(command[4]) / "DSC 0001.dng").write_bytes(b"DNG")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(raw_input.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="decode failed"):
        with raw_input.convert_nef_to_dng(nef, converter=executable) as dng:
            folder = dng.parent
            assert dng.is_file()
            raise RuntimeError("decode failed")
    assert not folder.exists()
    assert commands[0][:3] == [str(executable), "-c", "-p0"]
    assert commands[0][-1] == str(nef)
    assert nef.read_bytes() == b"test NEF"


@pytest.mark.parametrize("outcome", ["failure", "empty", "timeout"])
def test_converter_failures_clean_temporary_files(nef, monkeypatch, tmp_path, outcome):
    executable = tmp_path / "converter.exe"
    executable.write_bytes(b"fake")
    folders = []

    def run(command, **kwargs):
        folders.append(Path(command[4]))
        if outcome == "timeout":
            raise raw_input.subprocess.TimeoutExpired(command, 600)
        return SimpleNamespace(returncode=3 if outcome == "failure" else 0, stdout="", stderr="failed")

    monkeypatch.setattr(raw_input.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        with raw_input.convert_nef_to_dng(nef, converter=executable):
            pytest.fail("failed conversion must not yield")
    assert folders and not folders[0].exists()


def test_missing_converter_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        raw_input.find_dng_converter(tmp_path / "missing.exe")


def test_tiff_tags_include_raw_subifd(tmp_path):
    path = tmp_path / "mosaic.dng"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((2, 2, 3), np.uint8), photometric="rgb", subifds=1,
                     extratags=[(50721, "d", 9, tuple(NIKON_XYZ_TO_CAM.ravel()), False)])
        writer.write(np.zeros((6, 8), np.uint16), photometric=32803,
                     extratags=[(33422, "B", 4, b"\0\1\1\2", False)])
    result = frontend.dng_tags(str(path))
    assert result["ImageWidth"] == 2  # root preview metadata remains compatible
    assert result["_raw_ifd"]["ImageWidth"] == 8
    assert result["_raw_ifd"]["ImageLength"] == 6
    np.testing.assert_allclose(frontend._mat(result["ColorMatrix1"]), NIKON_XYZ_TO_CAM)


def test_legacy_b2y_delegates_decoded_raw(monkeypatch):
    values = (np.ones((6, 8, 3)), np.eye(3), frontend.RawInfo())
    monkeypatch.setattr(frontend, "load_dng", Mock(return_value=values))
    processing = Mock(return_value="result")
    monkeypatch.setattr(frontend, "b2y_from_raw", processing)
    assert frontend.b2y("old.dng", half=True, ev=.3) == "result"
    frontend.load_dng.assert_called_once_with("old.dng", half=True)
    processing.assert_called_once_with(*values, lux_index=None, cct=None, ev=.3, camera="main")


def test_load_dng_preserves_dual_matrix_and_half_option(monkeypatch):
    metadata = tags()
    metadata["ColorMatrix2"] = NIKON_XYZ_TO_CAM * 1.05
    metadata["CalibrationIlluminant1"] = 21
    metadata["CalibrationIlluminant2"] = 17
    monkeypatch.setattr(frontend, "dng_tags", lambda _: metadata)
    raw = FakeRaw()
    monkeypatch.setattr(frontend.rawpy, "imread", lambda _: raw)
    rgb, matrix, info = frontend.load_dng("old.dng", half=True)
    assert raw.kwargs["half_size"] is True
    assert rgb.dtype == np.float32 and matrix.dtype == np.float32
    assert info.extra["cct_source"] == "DNG dual-matrix AsShotNeutral"
    assert 1500 <= info.cct <= 20000


def test_installed_libraw_reads_embedded_xyz_to_camera_and_orientation(tmp_path):
    """Real LibRaw demosaic on a synthetic CFA TIFF/DNG, not real-camera proof."""
    path = tmp_path / "合成 RAW 方向.dng"
    matrix_rationals = tuple(v for x in NIKON_XYZ_TO_CAM.ravel() for v in (round(x * 10000), 10000))
    extra = [(271, "s", 0, "NIKON CORPORATION", False), (272, "s", 0, "NIKON Z 7_2", False),
             (274, "H", 1, 6, False), (50706, "B", 4, (1, 4, 0, 0), False),
             (50707, "B", 4, (1, 1, 0, 0), False), (50708, "s", 0, "Synthetic Z7II test", False),
             (33421, "H", 2, (2, 2), False), (33422, "B", 4, (0, 1, 1, 2), False),
             (50721, "2i", 9, matrix_rationals, False),
             (50728, "2I", 3, (1, 2, 1, 1, 7, 10), False), (50778, "H", 1, 21, False),
             (50717, "I", 1, 65535, False), (50714, "H", 1, 0, False),
             (50719, "I", 2, (0, 0), False), (50720, "I", 2, (128, 96), False)]
    tifffile.imwrite(path, np.full((96, 128), 10000, np.uint16), photometric=32803,
                     metadata=None, extratags=extra)
    with raw_input.rawpy.imread(str(path)) as raw:
        np.testing.assert_allclose(raw.rgb_xyz_matrix[:3], NIKON_XYZ_TO_CAM, atol=1e-6)
        assert raw.sizes.flip == 6
    rgb, matrix, info = frontend.load_dng(str(path))
    assert rgb.shape == (128, 96, 3) and np.isfinite(rgb).all()
    _, expected = frontend.camera_d50_transform(NIKON_XYZ_TO_CAM)
    np.testing.assert_allclose(matrix, expected, atol=1e-6)
    assert info.extra["orientation"] == 6
    # LibRaw0.21.4 exposes DefaultCrop as inset metadata; its default postprocess keeps ActiveArea.
    crop_path = tmp_path / "合成 RAW 默认裁切.dng"
    cropped_extra = [t for t in extra if t[0] not in (50719, 50720)]
    cropped_extra += [(50719, "I", 2, (3, 5), False), (50720, "I", 2, (120, 88), False),
                      (50829, "I", 4, (0, 0, 96, 128), False)]
    tifffile.imwrite(crop_path, np.arange(96 * 128, dtype=np.uint16).reshape(96, 128),
                     photometric=32803, metadata=None, extratags=cropped_extra)
    rgb, _, info = frontend.load_dng(str(crop_path))
    assert rgb.shape == (128, 96, 3)
    source_geometry = dict(info.extra, left_margin=3, top_margin=5, width=120, height=88)
    result = raw_input._reconcile_dng_dimensions(rgb, source_geometry, info, frontend.dng_tags(str(crop_path)))
    # Rotation6 is clockwise: source x3..123 -> output rows3..123; source y5..93 -> cols3..91.
    np.testing.assert_array_equal(result, rgb[3:123, 3:91])
