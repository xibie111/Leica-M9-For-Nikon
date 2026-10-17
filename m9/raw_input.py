"""Two explicit NEF routes into the same linear, D50-balanced camera space.

LibRaw's cam_xyz is XYZ -> camera, despite rawpy's ambiguous rgb_xyz_matrix
name. rawpy/_rawpy.pyx returns cam_xyz unchanged; LibRaw::cam_xyz_coeff in
src/utils/utils_dcraw.cpp multiplies it by xyz_rgb, normalizes, then inverts.
The Nikon Z 7 table entry in src/tables/colordata.cpp also covers Z 7_2.
For standard Z6_3 NEFs, the missing 0.21.4 camera-table entry is backported
from LibRaw 0.22.0 only when the installed library returns an all-zero matrix.
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import rawpy

from . import frontend
from .frontend import RawInfo


def _report(progress, stage: str, message: str):
    if progress is not None:
        progress(stage, message)


def find_dng_converter(explicit: str | Path | None = None) -> Path:
    """Resolve Adobe's installed converter; an explicit missing path is an error."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Adobe DNG Converter not found: {path}")
        return path.resolve()
    candidates = []
    if os.environ.get("M9_DNG_CONVERTER"):
        return find_dng_converter(os.environ["M9_DNG_CONVERTER"])
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(variable)
        if base:
            candidates.append(Path(base) / "Adobe" / "Adobe DNG Converter" / "Adobe DNG Converter.exe")
    command = shutil.which("Adobe DNG Converter.exe")
    if command:
        candidates.append(Path(command))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Adobe DNG Converter is required for method=dng. Install it or set M9_DNG_CONVERTER.")


@contextmanager
def convert_nef_to_dng(path: str | Path, converter: str | Path | None = None, progress=None):
    """Convert to a temporary lossless mosaic DNG; remove it even after errors."""
    executable = find_dng_converter(converter)
    with tempfile.TemporaryDirectory(prefix="m9-nef-") as folder:
        _report(progress, "convert", "Adobe DNG Converter: lossless compressed mosaic RAW")
        # -c is lossless compression, -p0 omits previews. Do not pass -l (linear DNG).
        command = [str(executable), "-c", "-p0", "-d", folder, str(Path(path).resolve())]
        try:
            result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=600,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Adobe DNG Converter timed out after 600 seconds") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not run Adobe DNG Converter: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "no diagnostic output").strip()
            raise RuntimeError(f"Adobe DNG Converter failed ({result.returncode}): {detail}")
        files = [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() == ".dng"]
        if len(files) != 1 or files[0].stat().st_size == 0:
            raise RuntimeError("Adobe DNG Converter did not produce exactly one nonempty DNG")
        yield files[0]


def _exposure_info(tags: dict) -> RawInfo:
    info = RawInfo(exposure_time=frontend._num(tags.get("ExposureTime")),
                   fnumber=frontend._num(tags.get("FNumber")),
                   iso=frontend._num(tags.get("ISOSpeedRatings") or tags.get("PhotographicSensitivity")),
                   baseline_exposure=frontend._num(tags.get("BaselineExposure")),
                   make=str(tags.get("Make", "")).strip(), model=str(tags.get("Model", "")).strip())
    for name in ("exposure_time", "fnumber", "iso"):
        value = getattr(info, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"NEF requires valid exposure metadata: {name}")
    if not np.isfinite(info.baseline_exposure):
        raise ValueError("RAW BaselineExposure is not finite")
    info.ev100 = float(np.log2(info.fnumber ** 2 / info.exposure_time) - np.log2(info.iso / 100.0))
    info.extra["orientation"] = frontend.orientation_from_tags(tags)
    for tag, name in (("ExposureTime", "exposure_time_rational"), ("ShutterSpeedValue", "shutter_speed_apex_rational")):
        value = tags.get(tag)
        if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, (int, np.integer)) for v in value):
            info.extra[name] = [int(value[0]), int(value[1])]
    return info


def _checked_rational(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2 or not all(isinstance(v, (int, np.integer)) for v in value) or value[1] == 0:
        raise ValueError(f"Exposure reconciliation requires exact valid {name} rational metadata")
    return Fraction(int(value[0]), int(value[1]))


def _check_apex(exposure, apex, stored_denominator):
    # APEX is rational/quantized; allow only half its stored least-significant step.
    if exposure <= 0 or abs(float(apex) + np.log2(float(exposure))) > 0.5 / abs(stored_denominator) + 1e-12:
        raise ValueError("DNG ShutterSpeedValue is inconsistent with ExposureTime at its rational precision")


def _prove_exposure_metadata_only(source_path, dng_path, source_info, tags):
    """Accept changed shutter metadata only after proving unchanged captured mosaic."""
    source_time = _checked_rational(source_info.extra.get("exposure_time_rational"), "source ExposureTime")
    converted_time = _checked_rational(tags.get("ExposureTime"), "DNG ExposureTime")
    apex = _checked_rational(tags.get("ShutterSpeedValue"), "DNG ShutterSpeedValue")
    _check_apex(converted_time, apex, int(tags["ShutterSpeedValue"][1]))
    if "shutter_speed_apex_rational" in source_info.extra:
        source_apex = source_info.extra["shutter_speed_apex_rational"]
        _check_apex(source_time, _checked_rational(source_apex, "source ShutterSpeedValue"), int(source_apex[1]))
    digest = hashlib.sha256()
    with rawpy.imread(str(source_path)) as source, rawpy.imread(str(dng_path)) as converted:
        a, b = source.raw_image, converted.raw_image
        if a.ndim != 2 or a.dtype != np.uint16 or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Exposure mismatch: source and DNG mosaic layouts differ")
        if not np.array_equal(source.raw_pattern, converted.raw_pattern) or source.black_level_per_channel != converted.black_level_per_channel:
            raise ValueError("Exposure mismatch: source and DNG CFA or black levels differ")
        for row in range(0, a.shape[0], 256):
            if not np.array_equal(a[row:row + 256], b[row:row + 256]):
                raise ValueError("Exposure mismatch: source and DNG captured mosaic pixels differ")
            digest.update(np.ascontiguousarray(a[row:row + 256]).data)
        shape = list(a.shape)
    return {"reason": "converted shutter metadata differs; identical full RAW mosaic and consistent converted APEX proven",
            "source_exposure_time_rational": list(source_info.extra["exposure_time_rational"]),
            "converted_exposure_time_rational": list(tags["ExposureTime"]),
            "converted_shutter_apex_rational": list(tags["ShutterSpeedValue"]),
            "mosaic_sha256": digest.hexdigest(), "mosaic_shape": shape,
            "m9_exposure_source": "original NEF ExposureTime, FNumber and ISO"}


def validate_converted_dng(path: str | Path, source_info: RawInfo | None = None, source_path=None) -> dict:
    """Reject rendered/linear, lossy, incomplete or exposure-changing DNG output."""
    tags = frontend.dng_tags(str(path))
    if "DNGVersion" not in tags:
        raise ValueError("Converter output has no DNGVersion")
    raw_tags = tags.get("_raw_ifd", tags)
    if int(frontend._num(raw_tags.get("PhotometricInterpretation"))) != 32803:
        raise ValueError("Converter output must retain a CFA mosaic RAW, not rendered RGB")
    if int(frontend._num(raw_tags.get("SamplesPerPixel", 1))) != 1 or "CFAPattern" not in raw_tags:
        raise ValueError("Converter output has no single-channel CFA mosaic")
    if int(frontend._num(raw_tags.get("Compression"))) != 7:
        raise ValueError("Converter output must use lossless JPEG RAW compression (DNG Compression=7)")
    if min(frontend._num(raw_tags.get("ImageWidth")), frontend._num(raw_tags.get("ImageLength"))) <= 0:
        raise ValueError("Converter output has invalid RAW dimensions")
    if frontend._mat(tags.get("ColorMatrix1")) is None:
        raise ValueError("Converter output has no ColorMatrix1")
    if "ColorMatrix2" in tags:
        frontend._mat(tags["ColorMatrix2"])
    if "AsShotNeutral" not in tags:
        raise ValueError("Converter output has no AsShotNeutral")
    neutral = frontend._rationals(tags["AsShotNeutral"])
    if neutral.shape != (3,) or not np.isfinite(neutral).all() or np.any(neutral <= 0):
        raise ValueError("Converter output has invalid AsShotNeutral")
    info = _exposure_info(tags)
    if source_info is not None:
        for name in ("fnumber", "iso"):
            if not np.isclose(getattr(source_info, name), getattr(info, name), rtol=0.002, atol=1e-9):
                raise ValueError(f"Converter changed exposure metadata: {name}")
        if not np.isclose(source_info.exposure_time, info.exposure_time, rtol=0.002, atol=1e-9):
            if source_path is None:
                raise ValueError("Converter changed exposure metadata: exposure_time")
            tags["_exposure_reconciliation"] = _prove_exposure_metadata_only(source_path, path, source_info, tags)
        if source_info.extra["orientation"] != info.extra["orientation"]:
            raise ValueError("Converter changed TIFF orientation")
    return tags


def _expected_dimensions(geometry: dict) -> tuple[int, int]:
    width, height = geometry["width"], geometry["height"]
    if min(width, height) <= 0:
        raise ValueError("LibRaw reports invalid active RAW dimensions")
    if geometry["flip"] & 4:
        width, height = height, width
    return width, height


def _check_dimensions(rgb: np.ndarray, geometry: dict):
    expected = _expected_dimensions(geometry)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or (rgb.shape[1], rgb.shape[0]) != expected:
        raise ValueError(f"Decoded image dimensions differ from oriented active RAW: expected {expected}, got {rgb.shape}")


def _photo_geometry(geometry: dict) -> dict:
    """Use LibRaw's camera inset crop, excluding border outside the intended photograph."""
    width, height = geometry.get("crop_width", 0), geometry.get("crop_height", 0)
    if width == 0 and height == 0:
        return dict(geometry)
    left, top = geometry.get("crop_left_margin", 0), geometry.get("crop_top_margin", 0)
    dx, dy = left - geometry["left_margin"], top - geometry["top_margin"]
    if min(width, height) <= 0 or min(dx, dy) < 0 or dx + width > geometry["width"] or dy + height > geometry["height"]:
        raise ValueError("LibRaw camera crop is invalid or outside the decoded active RAW")
    return dict(geometry, width=width, height=height, left_margin=left, top_margin=top,
                iwidth=width, iheight=height)


def _crop_camera_photo(rgb, geometry, target):
    _check_dimensions(rgb, geometry)
    dx, dy = target["left_margin"] - geometry["left_margin"], target["top_margin"] - geometry["top_margin"]
    if dx == 0 and dy == 0 and geometry["width"] == target["width"] and geometry["height"] == target["height"]:
        return rgb
    sensor = _orient_pixels(rgb, geometry["flip"], inverse=True)
    sensor = sensor[dy:dy + target["height"], dx:dx + target["width"]]
    result = _orient_pixels(sensor, target["flip"]).copy()
    _check_dimensions(result, target)
    return result


def _integer_pair(value, name: str) -> tuple[int, int]:
    values = np.asarray(value, np.float64).ravel()
    if values.size == 4:
        if np.any(values[1::2] == 0):
            raise ValueError(f"DNG {name} has a zero rational denominator")
        values = values[0::2] / values[1::2]
    if values.shape != (2,) or not np.isfinite(values).all() or not np.equal(values, np.round(values)).all():
        raise ValueError(f"DNG {name} must specify two exact integer pixel coordinates")
    return int(values[0]), int(values[1])


def _orient_pixels(rgb, flip, inverse=False):
    """LibRaw's transpose/row-reverse/column-reverse flags, with no resampling."""
    if inverse and flip & 4:
        rgb = rgb.swapaxes(0, 1)
    if flip & 1:
        rgb = rgb[:, ::-1]
    if flip & 2:
        rgb = rgb[::-1]
    if not inverse and flip & 4:
        rgb = rgb.swapaxes(0, 1)
    return rgb


def _reconcile_dng_dimensions(rgb, source_geometry, info, tags):
    if rgb.ndim == 3 and rgb.shape[2] == 3 and (rgb.shape[1], rgb.shape[0]) == _expected_dimensions(source_geometry):
        return rgb
    geometry = info.extra
    raw_tags = tags.get("_raw_ifd", tags)
    missing = [key for key in ("raw_width", "raw_height", "left_margin", "top_margin", "width", "height", "flip")
               if key not in geometry]
    if not missing and "DefaultCropOrigin" in raw_tags and "DefaultCropSize" in raw_tags:
        origin = _integer_pair(raw_tags["DefaultCropOrigin"], "DefaultCropOrigin")
        size = _integer_pair(raw_tags["DefaultCropSize"], "DefaultCropSize")
        left = geometry["left_margin"] + origin[0]
        top = geometry["top_margin"] + origin[1]
        # Reconcile only the same sensor coordinate system and the source's exact active rectangle.
        same_sensor = all(geometry[k] == source_geometry[k] for k in ("raw_width", "raw_height", "flip"))
        same_crop = (left, top, *size) == (source_geometry["left_margin"], source_geometry["top_margin"],
                                        source_geometry["width"], source_geometry["height"])
        contained = min(origin) >= 0 and origin[0] + size[0] <= geometry["width"] and origin[1] + size[1] <= geometry["height"]
        if same_sensor and same_crop and contained:
            _check_dimensions(rgb, geometry)
            sensor = _orient_pixels(rgb, geometry["flip"], inverse=True)
            sensor = sensor[origin[1]:origin[1] + size[1], origin[0]:origin[0] + size[0]]
            rgb = _orient_pixels(sensor, source_geometry["flip"]).copy()
            _check_dimensions(rgb, source_geometry)
            info.extra["dng_crop_reconciliation"] = {"source_sensor_rectangle": [left, top, *size],
                                                      "reason": "DNG DefaultCrop exactly matches all source active pixels; no resampling"}
            info.extra["decoded_dimensions"] = [int(rgb.shape[1]), int(rgb.shape[0])]
            return rgb
    raise ValueError("Converted DNG does not preserve the source active dimensions and no exact integer crop mapping was proven. "
                     "No pixels were resized or discarded to force a match. Choose native explicitly or check converter crop settings.")


def _native_xyz2cam(path: Path, info: RawInfo, matrix, progress=None):
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (4, 3) or not np.allclose(matrix[3], 0, atol=1e-6):
        raise ValueError("Unsupported LibRaw colour matrix layout")
    info.extra["matrix_source"] = "LibRaw cam_xyz (XYZ -> camera), single matrix"
    if (not np.any(matrix) and info.model.strip().upper() == "NIKON Z6_3"
            and info.make.strip().upper() in ("NIKON", "NIKON CORPORATION")):
        tags = frontend.dng_tags(str(path))
        raw_tags = tags.get("_raw_ifd", tags)
        offsets = np.asarray(raw_tags.get("StripOffsets", ()), np.float64).ravel()
        if (int(frontend._num(raw_tags.get("Compression"))) != 34713
                or int(frontend._num(raw_tags.get("PhotometricInterpretation"))) != 32803
                or int(frontend._num(raw_tags.get("SamplesPerPixel", 1))) != 1
                or offsets.shape != (1,) or not np.isfinite(offsets).all()
                or offsets[0] < 0 or offsets[0] != int(offsets[0])):
            raise ValueError("Z6 III matrix compatibility requires a standard compressed single-strip CFA NEF")
        with path.open("rb") as stream:
            stream.seek(int(offsets[0]))
            signature = stream.read(4)
        if len(signature) != 4 or signature == b"\xff\x10\xff\x50":
            raise ValueError("Z6 III HE/HE* or incomplete RAW data is not supported by this matrix compatibility path")
        # Exact Nikon Z6_3 entry, not Z6/Z6_2. Match LibRaw's float / 10000.f.
        # https://github.com/LibRaw/LibRaw/blob/0.22.0/src/tables/colordata.cpp#L1036-L1037
        xyz2cam = np.array([[11206, -4286, -941], [-4879, 12847, 2251],
                            [-745, 1654, 7374]], dtype=np.float32) / np.float32(10000)
        info.extra["matrix_source"] = "LibRaw 0.22.0 Nikon Z6_3 camera table (XYZ -> camera), missing-profile backport"
        info.extra["matrix_compatibility_profile"] = {
            "camera_model": "NIKON Z6_3", "upstream_libraw_version": "0.22.0",
            "source_url": "https://github.com/LibRaw/LibRaw/blob/0.22.0/src/tables/colordata.cpp#L1036-L1037",
            "reason": "installed LibRaw returned an all-zero cam_xyz",
            "xyz_to_camera": xyz2cam.tolist(), "raw_strip_signature": signature.hex(),
        }
        _report(progress, "metadata", "LibRaw has no Nikon Z6_3 colour matrix; using the verified LibRaw 0.22.0 camera table on the native route")
        return xyz2cam.astype(np.float64)
    return matrix[:3]


def _native_nef(path: Path, info: RawInfo, progress=None):
    _report(progress, "decode", "LibRaw native NEF: full-size demosaic and locked D50 white balance")
    with rawpy.imread(str(path)) as raw:
        geometry = frontend.raw_geometry(raw)
        if raw.num_colors != 3 or bytes(raw.color_desc[:3]) != b"RGB":
            raise ValueError("Native NEF route requires a three-colour RGB Bayer sensor")
        pattern = raw.raw_pattern
        if pattern is None or np.asarray(pattern).shape != (2, 2):
            raise ValueError("Native NEF route requires a Bayer CFA mosaic")
        # These rows are XYZ -> camera, not camera -> XYZ. See module source note.
        xyz2cam = _native_xyz2cam(path, info, raw.rgb_xyz_matrix, progress)
        mul, cam2srgb = frontend.camera_d50_transform(xyz2cam)
        wb = np.asarray(raw.camera_whitebalance[:3], np.float64)
        if wb.shape != (3,) or not np.isfinite(wb).all() or np.any(wb <= 0):
            raise ValueError("NEF has no valid as-shot white balance")
        neutral = wb[1] / wb
        xyz = np.linalg.solve(xyz2cam, neutral)
        info.cct = frontend.scene_cct_from_xyz(xyz)
        info.as_shot_neutral = neutral
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16,
                              user_wb=[mul[0], mul[1], mul[2], mul[1]], output_color=rawpy.ColorSpace.raw,
                              demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, half_size=False,
                              highlight_mode=rawpy.HighlightMode.Clip)
    photo_geometry = _photo_geometry(geometry)
    rgb = _crop_camera_photo(rgb, geometry, photo_geometry)
    info.extra.update(geometry)
    info.extra.update(method="native", decoded_dimensions=[int(rgb.shape[1]), int(rgb.shape[0])],
                      source_camera_rectangle=[photo_geometry[k] for k in ("left_margin", "top_margin", "width", "height")],
                      cct_source="approximate: LibRaw single matrix + as-shot neutral (McCamy)",
                      cct_is_approximate=True, libraw_version=list(rawpy.libraw_version), rawpy_version=rawpy.__version__)
    return rgb.astype(np.float32) / 65535.0, cam2srgb, info


def load_nef(path: str | Path, method: str = "dng", converter: str | Path | None = None, progress=None):
    """Return linear D50 camera RGB, camera-to-sRGB, RawInfo; never switch routes."""
    path = Path(path).expanduser().resolve()
    if method not in ("dng", "native"):
        raise ValueError("method must be 'dng' or 'native'")
    if path.suffix.lower() != ".nef":
        raise ValueError("Input must be a Nikon .NEF file")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"NEF input is missing or empty: {path}")
    _report(progress, "metadata", "Read NEF exposure, orientation and camera metadata")
    try:
        source_info = _exposure_info(frontend.dng_tags(str(path)))
    except Exception as exc:
        raise ValueError(f"Cannot read NEF metadata: {exc}") from exc
    if "NIKON" not in source_info.make.upper():
        raise ValueError(f"Input does not identify a Nikon camera: {source_info.make!r}")
    if method == "native":
        return _native_nef(path, source_info, progress)
    # Inspect LibRaw's visible active area before conversion, without rendering RGB.
    with rawpy.imread(str(path)) as raw:
        source_geometry = frontend.raw_geometry(raw)
    photo_geometry = _photo_geometry(source_geometry)
    with convert_nef_to_dng(path, converter=converter, progress=progress) as converted:
        _report(progress, "validate", "Check lossless CFA DNG, colour matrices, exposure and dimensions")
        tags = validate_converted_dng(converted, source_info, source_path=path)
        _report(progress, "decode", "Decode temporary DNG through the existing dual-matrix front end")
        rgb, cam2srgb, info = frontend.load_dng(str(converted))
        rgb = _reconcile_dng_dimensions(rgb, photo_geometry, info, tags)
        for name in ("exposure_time", "fnumber", "iso"):
            if name == "exposure_time" and "_exposure_reconciliation" in tags:
                if not np.isclose(info.exposure_time, frontend._num(tags["ExposureTime"]), rtol=0, atol=1e-12):
                    raise ValueError("Decoded DNG exposure disagrees with validated DNG metadata")
                continue
            if not np.isclose(getattr(source_info, name), getattr(info, name), rtol=0.002, atol=1e-9):
                raise ValueError(f"Decoded DNG changed exposure metadata: {name}")
        if "_exposure_reconciliation" in tags:
            info.extra["exposure_metadata_reconciliation"] = tags["_exposure_reconciliation"]
        info.extra["converted_exposure_seconds"] = info.exposure_time
        if "exposure_time_rational" in source_info.extra:
            info.extra["exposure_time_rational"] = source_info.extra["exposure_time_rational"]
        for name in ("exposure_time", "fnumber", "iso", "ev100"):
            setattr(info, name, getattr(source_info, name))
        info.extra.update(method="dng", converter=str(find_dng_converter(converter)),
                          source_active_dimensions=list(_expected_dimensions(photo_geometry)),
                          source_camera_rectangle=[photo_geometry[k] for k in ("left_margin", "top_margin", "width", "height")],
                          dng_compression=int(frontend._num(tags.get("_raw_ifd", tags)["Compression"])),
                          rawpy_version=rawpy.__version__, libraw_version=list(rawpy.libraw_version))
        return rgb, cam2srgb, info
