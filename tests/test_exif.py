from io import BytesIO
import json
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import tifffile

from m9 import exif
from m9.uhdr import encode_p3_jpeg, icc_p3


@pytest.fixture
def jpeg():
    return encode_p3_jpeg(np.random.default_rng(19).random((19, 31, 3), dtype=np.float32), 95)


@pytest.fixture
def metadata(monkeypatch):
    values = {
        "Make": "NIKON CORPORATION", "Model": "NIKON Z6_3",
        "DateTime": "2026:10:07 15:53:41", "DateTimeOriginal": "2026:10:07 15:53:41",
        "DateTimeDigitized": "2026:10:07 15:53:41", "OffsetTime": "+08:00",
        "OffsetTimeOriginal": "+08:00", "OffsetTimeDigitized": "+08:00",
        "SubSecTime": "123", "SubSecTimeOriginal": "123", "SubSecTimeDigitized": "123",
        "ExposureTime": (10, 6400), "FNumber": (630, 100), "ISOSpeedRatings": 100,
        "ExposureBiasValue": (-2, 6), "FocalLength": (2000, 10), "FocalLengthIn35mmFilm": 300,
        "LensSpecification": (240, 10, 2000, 10, 400, 100, 630, 100),
        "LensMake": "NIKON", "LensModel": "NIKKOR Z 24-200mm f/4-6.3 VR",
        "GPSInfo": {1: "N"}, "MakerNote": b"private maker notes", "BodySerialNumber": "private serial",
        "LensSerialNumber": "private lens serial", "Artist": "private owner", "Orientation": 8,
    }
    read = Mock(return_value=(values, {}))
    monkeypatch.setattr(exif, "_read_nef_fields", read)
    return values, read


def _read(jpeg):
    with Image.open(BytesIO(jpeg)) as image:
        root = image.getexif()
        nested = root.get_ifd(34665)
        return dict(root), nested, image.info["exif"]


def _type_codes(payload):
    tiff = payload[6:]
    root_offset, = struct.unpack_from("<I", tiff, 4)

    def directory(offset):
        count, = struct.unpack_from("<H", tiff, offset)
        return {tag: (kind, value) for tag, kind, _, value in (
            struct.unpack_from("<HHII", tiff, offset + 2 + 12 * index) for index in range(count))}

    root = directory(root_offset)
    return root, directory(root[34665][1])


def test_roundtrip_white_list_signed_rationals_orientation_dimensions_and_p3(jpeg, metadata):
    values, read = metadata
    source = Path("original NEF.NEF")
    output, summary = exif.add_nef_exif(jpeg, source, (31, 19))
    read.assert_called_once_with(source)
    root, nested, payload = _read(output)
    assert root[271] == values["Make"] and root[272] == values["Model"]
    assert root[306] == values["DateTime"]
    assert (root[274], root[256], root[257]) == (1, 31, 19)
    assert (nested[40961], nested[40962], nested[40963]) == (65535, 31, 19)
    assert nested[33434] == 1 / 640 and nested[33437] == 6.3 and nested[34855] == 100
    assert nested[37380] == -1 / 3 and nested[37386] == 200 and nested[41989] == 300
    assert nested[42034] == (24, 200, 4, 6.3)
    assert nested[42035] == values["LensMake"] and nested[42036] == values["LensModel"]
    assert nested[36867] == values["DateTimeOriginal"] and nested[36868] == values["DateTimeDigitized"]
    assert nested[36880] == nested[36881] == nested[36882] == "+08:00"
    assert nested[37520] == nested[37521] == nested[37522] == "123"
    assert 34853 not in root and not ({37500, 42032, 42033, 42037, 513, 514} & (root.keys() | nested.keys()))
    assert b"private" not in payload and "Artist" not in summary["retained"]
    assert summary["success"] and summary["source_read"] and summary["source"] == str(source)
    assert set(summary["retained"]) == set(exif._FIELDS)
    assert json.loads(json.dumps(summary)) == summary
    root_types, nested_types = _type_codes(payload)
    assert root_types[274][0] == 3 and root_types[256][0] == 4
    assert nested_types[37380][0] == 10 and nested_types[33434][0] == 5
    assert nested_types[34855][0] == nested_types[41989][0] == nested_types[40961][0] == 3


def test_app1_insertion_preserves_all_compressed_bytes_pixels_icc_and_no_sidecars(jpeg, metadata, tmp_path):
    source = tmp_path / "source.NEF"
    source.write_bytes(b"synthetic source is not rewritten")
    before = source.read_bytes()
    output, _ = exif.add_nef_exif(jpeg, source, (31, 19))
    assert output[:4] == b"\xff\xd8\xff\xe1"
    segment_length = int.from_bytes(output[4:6], "big")
    assert output[2 + 2 + segment_length:] == jpeg[2:]
    assert exif._without_exif(output) == jpeg
    with Image.open(BytesIO(jpeg)) as old, Image.open(BytesIO(output)) as new:
        np.testing.assert_array_equal(np.asarray(old), np.asarray(new))
        assert old.info["icc_profile"] == new.info["icc_profile"] == icc_p3()
    assert source.read_bytes() == before and list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("value", [(-2, 6), (0, 6), (2, 6), .5])
def test_exposure_compensation_is_always_signed_metadata_only(jpeg, metadata, value):
    values, _ = metadata
    values["ExposureBiasValue"] = value
    output, _ = exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))
    _, nested, payload = _read(output)
    expected = value[0] / value[1] if isinstance(value, tuple) else value
    assert nested[37380] == expected
    assert _type_codes(payload)[1][37380][0] == 10
    assert exif._without_exif(output) == jpeg


@pytest.mark.parametrize("name,value", [
    ("Model", "x" * 257), ("Make", b"NIKON"), ("LensModel", "镜头"), ("LensMake", "bad\ntext"),
    ("DateTimeOriginal", "2026:02:30 01:02:03"), ("DateTime", "2026-01-01 00:00:00"),
    ("OffsetTime", "+24:00"), ("SubSecTime", "1.2"), ("SubSecTimeOriginal", "1" * 33),
    ("ExposureTime", (1, 0)), ("ExposureTime", 0), ("FNumber", -2), ("FNumber", True),
    ("FocalLength", np.nan), ("FocalLength", np.inf), ("ExposureBiasValue", (2**40, 1)),
    ("ISOSpeedRatings", 100.5), ("ISOSpeedRatings", True), ("ISOSpeedRatings", 0),
    ("FocalLengthIn35mmFilm", 65536), ("LensSpecification", (24, 200, 4)),
    ("LensSpecification", (24, 200, -4, 6.3)), ("ExposureTime", (1.5, 2)),
])
def test_bad_optional_value_is_skipped_without_losing_other_fields(jpeg, metadata, name, value):
    values, _ = metadata
    values[name] = value
    output, summary = exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))
    root, nested, _ = _read(output)
    assert summary["success"] and name in summary["skipped"] and name not in summary["retained"]
    assert exif._FIELDS[name][0] not in (root if exif._FIELDS[name][1] == 0 else nested)
    assert root[274] == 1 and nested[40962] == 31 and nested[40961] == 65535
    assert exif._without_exif(output) == jpeg


def test_missing_metadata_still_has_valid_basic_exif(jpeg, metadata):
    values, _ = metadata
    values.clear()
    output, summary = exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))
    root, nested, _ = _read(output)
    assert summary["success"] and summary["source_read"] and not summary["retained"]
    assert set(summary["skipped"]) == set(exif._FIELDS)
    assert root[274] == 1 and nested[40962] == 31 and nested[40963] == 19


@pytest.mark.parametrize("failure", [OSError("unavailable"), ValueError("damaged metadata"), RuntimeError("reader failed")])
def test_optional_reader_failure_does_not_fail_the_jpeg(jpeg, metadata, failure):
    _, reader = metadata
    reader.side_effect = failure
    output, summary = exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))
    assert summary["success"] and not summary["source_read"]
    assert "source_metadata" in summary["skipped"]
    assert _read(output)[0][274] == 1


@pytest.mark.parametrize("failure", [MemoryError(), KeyboardInterrupt()])
@pytest.mark.parametrize("stage", ["read", "write"])
def test_resource_and_interrupt_exceptions_are_not_hidden(jpeg, metadata, monkeypatch, failure, stage):
    if stage == "read":
        metadata[1].side_effect = failure
    else:
        monkeypatch.setattr(exif, "_payload", Mock(side_effect=failure))
    with pytest.raises(type(failure)):
        exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))


@pytest.mark.parametrize("iso", [65535, 65536, 128000, 2**32 - 1])
def test_high_iso_uses_legal_short_sentinel_and_long_iso_speed(jpeg, metadata, iso):
    metadata[0]["ISOSpeedRatings"] = iso
    output, _ = exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))
    _, nested, payload = _read(output)
    assert nested[34855] == min(iso, 65535)
    if iso > 65535:
        assert nested[34864] == 3 and nested[34867] == iso
        assert _type_codes(payload)[1][34867][0] == 4
    assert _type_codes(payload)[1][34855][0] == 3


def test_app1_upper_bound_falls_back_to_basic_metadata(jpeg, metadata, monkeypatch):
    actual = exif._payload
    monkeypatch.setattr(exif, "_payload", lambda fields, size: b"x" * 65534 if fields else actual(fields, size))
    output, summary = exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))
    assert summary["success"] and not summary["retained"]
    assert all(name in summary["skipped"] for name in exif._FIELDS)
    assert len(_read(output)[2]) <= 65533 and exif._without_exif(output) == jpeg


def test_app1_limit_accepts_exact_maximum_and_rejects_even_basic_oversize(jpeg, metadata, monkeypatch):
    basic = exif._payload({}, (31, 19))
    monkeypatch.setattr(exif, "_payload", lambda fields, size: basic + b"\0" * (65533 - len(basic)))
    output, summary = exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))
    assert summary["success"] and int.from_bytes(output[4:6], "big") == 65535
    assert _read(output)[0][274] == 1
    monkeypatch.setattr(exif, "_payload", lambda fields, size: b"x" * 65534)
    unchanged, summary = exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))
    assert unchanged == jpeg and not summary["success"] and "basic EXIF" in summary["reason"]


def test_serialization_failure_uses_valid_basic_exif(jpeg, metadata, monkeypatch):
    actual = exif._payload
    def serialize(fields, size):
        if fields:
            raise struct.error("invalid optional field")
        return actual(fields, size)
    monkeypatch.setattr(exif, "_payload", serialize)
    output, summary = exif.add_nef_exif(jpeg, Path("source.NEF"), (31, 19))
    assert summary["success"] and not summary["retained"] and _read(output)[0][274] == 1


def test_replaces_old_exif_without_changing_xmp_icc_or_scan(jpeg, metadata):
    old = Image.Exif()
    old[274] = 8
    old[34853] = {1: "N", 2: (1, 2, 3)}
    payload = old.tobytes()
    xmp = b"http://ns.adobe.com/xap/1.0/\0preserved XMP"
    before = jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(xmp) + 2) + xmp + jpeg[2:]
    tagged = before[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + before[2:]
    output, _ = exif.add_nef_exif(tagged, Path("source.NEF"), (31, 19))
    assert output.count(b"Exif\0\0") == 1 and exif._without_exif(output) == before
    assert 34853 not in _read(output)[0]


@pytest.mark.parametrize("size", [(0, 19), (31, -1), (31.5, 19), (True, 19), (31,), (2**32, 19), None])
def test_invalid_output_dimensions_are_programming_errors(jpeg, metadata, size):
    with pytest.raises(ValueError, match="dimensions"):
        exif.add_nef_exif(jpeg, Path("source.NEF"), size)


@pytest.mark.parametrize("bad", [b"not JPEG", b"\xff\xd8\xff", b"\xff\xd8\xff\xe2\x00\x20short"])
def test_invalid_encoded_jpeg_is_returned_with_failure_summary(metadata, bad):
    output, summary = exif.add_nef_exif(bad, Path("source.NEF"), (31, 19))
    assert output == bad and not summary["success"]


def test_reader_filters_root_and_exif_without_loading_pixels(tmp_path, monkeypatch):
    source = tmp_path / "source.NEF"
    # A complete tiny TIFF exercises the real reader without a user RAW fixture.
    fields = Image.Exif()
    fields[271], fields[272] = "NIKON", "NIKON Z6_3"
    fields[34665] = {33434: exif._rational((1, 640)), 33437: exif._rational((63, 10)),
                    34855: 100, 37520: "73", 37521: "73", 37522: "73"}
    Image.new("L", (1, 1)).save(source, format="TIFF", exif=fields.tobytes())
    pixels = Mock(side_effect=AssertionError("metadata reader must not decode pixels"))
    monkeypatch.setattr(tifffile.TiffPage, "asarray", pixels)
    values, skipped = exif._read_nef_fields(source)
    assert values["Make"] == "NIKON" and values["Model"] == "NIKON Z6_3"
    assert values["ExposureTime"] == (1, 640) and values["FNumber"] == (63, 10)
    assert values["ISOSpeedRatings"] == 100 and not skipped
    assert values["SubSecTime"] == values["SubSecTimeOriginal"] == values["SubSecTimeDigitized"] == "73"
    pixels.assert_not_called()


def test_reader_skips_broken_optional_tag_without_touching_excluded_tags(monkeypatch):
    good = SimpleNamespace(count=5, value="NIKON")
    class Broken:
        count = 12
        @property
        def value(self):
            raise ValueError("broken")
    excluded = Mock()
    tags = {271: good, 272: Broken(), 34665: SimpleNamespace(value={"LensModel": "Lens", "MakerNote": excluded}),
            34853: excluded, 42033: excluded}
    context = Mock()
    context.__enter__ = Mock(return_value=SimpleNamespace(pages=[SimpleNamespace(tags=tags)]))
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(exif.tifffile, "TiffFile", Mock(return_value=context))
    values, skipped = exif._read_nef_fields(Path("source.NEF"))
    assert values == {"Make": "NIKON", "LensModel": "Lens"} and "Model" in skipped
    assert not excluded.mock_calls
