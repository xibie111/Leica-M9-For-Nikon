"""Optional, privacy-limited NEF EXIF for an already encoded Display P3 JPEG."""
from __future__ import annotations

from datetime import datetime
from fractions import Fraction
import math
from numbers import Integral, Real
from pathlib import Path
import re
import struct

from PIL.TiffImagePlugin import IFDRational, ImageFileDirectory_v2
import tifffile


# (TIFF tag, IFD0 or ExifIFD, value kind). No GPS, IDs, MakerNote or thumbnail.
_FIELDS = {
    "Make": (271, 0, "text"), "Model": (272, 0, "text"), "DateTime": (306, 0, "date"),
    "ExposureTime": (33434, 1, "positive"), "FNumber": (33437, 1, "positive"),
    "ISOSpeedRatings": (34855, 1, "iso"),
    "DateTimeOriginal": (36867, 1, "date"), "DateTimeDigitized": (36868, 1, "date"),
    "OffsetTime": (36880, 1, "offset"), "OffsetTimeOriginal": (36881, 1, "offset"),
    "OffsetTimeDigitized": (36882, 1, "offset"),
    "ExposureBiasValue": (37380, 1, "signed"), "FocalLength": (37386, 1, "positive"),
    "SubSecTime": (37520, 1, "subsecond"), "SubSecTimeOriginal": (37521, 1, "subsecond"),
    "SubSecTimeDigitized": (37522, 1, "subsecond"),
    "FocalLengthIn35mmFilm": (41989, 1, "short"),
    "LensSpecification": (42034, 1, "lens"), "LensMake": (42035, 1, "text"),
    "LensModel": (42036, 1, "text"),
}
_MAX_APP1 = 65533  # JPEG's two-byte segment length includes its own two bytes.


def _read_nef_fields(source: Path):
    """Read only the root camera metadata and its standard Exif IFD; never pixels."""
    values, skipped = {}, {}
    with tifffile.TiffFile(source) as raw:
        tags = raw.pages[0].tags
        for name, (tag_id, _, _) in _FIELDS.items():
            tag = tags.get(tag_id)
            if tag is not None:
                try:
                    if tag.count > 4096:
                        raise ValueError("metadata value is too long")
                    values[name] = tag.value
                except MemoryError:
                    raise
                except Exception as error:
                    skipped[name] = f"read failed: {type(error).__name__}"
        exif_tag = tags.get(34665)
        if exif_tag is not None:
            try:
                exif = exif_tag.value
                if not isinstance(exif, dict):
                    raise ValueError("invalid Exif IFD")
                for name, (tag_id, _, _) in _FIELDS.items():
                    # tifffile spells SubsecTime differently from Pillow/ExifTags.
                    source_name = tifffile.TIFF.TAGS.get(tag_id, name)
                    if source_name in exif:
                        values.setdefault(name, exif[source_name])
                if values.get("ISOSpeedRatings") == 65535 and "ISOSpeed" in exif:
                    values["ISOSpeedRatings"] = exif["ISOSpeed"]
            except MemoryError:
                raise
            except Exception as error:
                skipped["ExifIFD"] = f"read failed: {type(error).__name__}"
    return values, skipped


def _rational(value, signed=False, allow_zero=False):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if not all(isinstance(part, Integral) and not isinstance(part, bool) for part in value):
            raise ValueError("invalid rational components")
        fraction = Fraction(int(value[0]), int(value[1]))
    elif isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        fraction = Fraction(value).limit_denominator(1_000_000)
    else:
        raise ValueError("invalid finite rational")
    maximum = 2**31 - 1 if signed else 2**32 - 1
    if (abs(fraction.numerator) > maximum or fraction.denominator > maximum
            or (not signed and (fraction < 0 or (not allow_zero and fraction == 0)))):
        raise ValueError("rational outside EXIF range")
    return IFDRational(fraction.numerator, fraction.denominator)


def _validate(value, kind):
    if kind in ("positive", "signed"):
        return _rational(value, signed=kind == "signed")
    if kind == "lens":
        if not isinstance(value, (tuple, list)) or len(value) not in (4, 8):
            raise ValueError("expected four lens rationals")
        parts = list(zip(value[::2], value[1::2])) if len(value) == 8 else value
        return tuple(_rational(part, allow_zero=True) for part in parts)
    if kind in ("iso", "short"):
        maximum = 2**32 - 1 if kind == "iso" else 65535
        if not isinstance(value, Integral) or isinstance(value, bool) or not 0 < value <= maximum:
            raise ValueError("invalid EXIF integer")
        return int(value)
    if not isinstance(value, str):
        raise ValueError("expected ASCII text")
    value = value.rstrip("\0").strip()
    limit = {"date": 19, "offset": 6, "subsecond": 32}.get(kind, 256)
    if not value or len(value) > limit or any(not 32 <= ord(char) <= 126 for char in value):
        raise ValueError("empty, non-ASCII, control or overlong text")
    if kind == "date":
        if not re.fullmatch(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", value):
            raise ValueError("invalid EXIF date")
        datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    elif kind == "offset" and not re.fullmatch(r"[+-](?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("invalid UTC offset")
    elif kind == "subsecond" and not value.isdigit():
        raise ValueError("invalid subsecond digits")
    return value


def _payload(fields: dict, size: tuple[int, int]) -> bytes:
    root, exif = ImageFileDirectory_v2(), ImageFileDirectory_v2()

    def put(ifd, tag, value, field_type):
        ifd.tagtype[tag] = field_type
        ifd[tag] = value

    width, height = size
    for tag, value, field_type in ((256, width, 4), (257, height, 4), (274, 1, 3), (34665, 0, 4)):
        put(root, tag, value, field_type)
    # CIPA Exif 2.3: non-sRGB uses Uncalibrated; the existing ICC defines Display P3.
    for tag, value, field_type in ((36864, b"0230", 7), (40960, b"0100", 7),
                                   (40961, 65535, 3), (40962, width, 4), (40963, height, 4)):
        put(exif, tag, value, field_type)
    for name, value in fields.items():
        tag, nested, kind = _FIELDS[name]
        if kind == "iso" and value > 65535:
            put(exif, 34855, 65535, 3)
            put(exif, 34864, 3, 3)  # ISO speed, with its LONG value in ISOSpeed.
            put(exif, 34867, value, 4)
            continue
        field_type = {"signed": 10, "positive": 5, "lens": 5, "iso": 3, "short": 3}.get(kind, 2)
        put(exif if nested else root, tag, value, field_type)
    exif_offset = 8 + len(root.tobytes(8))
    root[34665] = exif_offset
    return b"Exif\0\0II*\0\x08\0\0\0" + root.tobytes(8) + exif.tobytes(exif_offset)


def _without_exif(jpeg: bytes) -> bytes:
    """Remove only existing EXIF APP1 segments; retain the encoded image verbatim."""
    if not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG SOI")
    parts, offset = [jpeg[:2]], 2
    while offset < len(jpeg):
        start = offset
        if jpeg[offset] != 255:
            raise ValueError("invalid JPEG marker")
        while offset < len(jpeg) and jpeg[offset] == 255:
            offset += 1
        if offset >= len(jpeg):
            raise ValueError("truncated JPEG marker")
        marker = jpeg[offset]
        offset += 1
        if marker in (0xDA, 0xD9):
            return b"".join(parts) + jpeg[start:]
        if marker == 1 or 0xD0 <= marker <= 0xD7:
            parts.append(jpeg[start:offset])
            continue
        length = int.from_bytes(jpeg[offset:offset + 2], "big")
        if length < 2 or offset + length > len(jpeg):
            raise ValueError("invalid JPEG segment length")
        end = offset + length
        if marker != 0xE1 or jpeg[offset + 2:offset + 8] != b"Exif\0\0":
            parts.append(jpeg[start:end])
        offset = end
    raise ValueError("JPEG has no image scan or EOI")


def add_nef_exif(jpeg: bytes, source: Path, size: tuple[int, int]) -> tuple[bytes, dict]:
    """Add optional source EXIF; never re-encode pixels or change the existing ICC."""
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(not isinstance(v, Integral) or isinstance(v, bool) or not 0 < v < 2**32 for v in size)):
        raise ValueError("EXIF dimensions must be two positive 32-bit integers")
    size = tuple(int(value) for value in size)
    summary = {"source": str(source), "success": False, "source_read": False,
               "retained": [], "skipped": {},
               "normalized": {"Orientation": 1, "ImageWidth": size[0], "ImageLength": size[1],
                              "PixelXDimension": size[0], "PixelYDimension": size[1], "ColorSpace": 65535}}
    try:
        clean = _without_exif(jpeg)
    except (ValueError, TypeError) as error:
        summary["reason"] = str(error)
        return jpeg, summary
    try:
        values, skipped = _read_nef_fields(Path(source))
        summary["source_read"] = True
        summary["skipped"].update(skipped)
    except MemoryError:
        raise
    except Exception as error:
        values = {}
        summary["skipped"]["source_metadata"] = f"read failed: {type(error).__name__}"
    fields = {}
    for name, (_, _, kind) in _FIELDS.items():
        if name not in values:
            summary["skipped"].setdefault(name, "missing")
            continue
        try:
            fields[name] = _validate(values[name], kind)
            summary["skipped"].pop(name, None)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
            summary["skipped"][name] = str(error)
    try:
        payload = _payload(fields, size)
        if len(payload) > _MAX_APP1:
            raise ValueError("EXIF APP1 payload exceeds 65533 bytes")
    except MemoryError:
        raise
    except Exception as error:
        reason = f"serialization failed: {type(error).__name__}: {str(error)[:160]}"
        for name in fields:
            summary["skipped"][name] = reason
        fields = {}
        try:
            payload = _payload({}, size)
            if len(payload) > _MAX_APP1:
                raise ValueError("basic EXIF APP1 payload exceeds limit")
        except MemoryError:
            raise
        except Exception as basic_error:
            summary["reason"] = f"basic EXIF unavailable: {type(basic_error).__name__}"
            return jpeg, summary
    summary["retained"] = list(fields)
    summary["success"] = True
    return clean[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + clean[2:], summary
