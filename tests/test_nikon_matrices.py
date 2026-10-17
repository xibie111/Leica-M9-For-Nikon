import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from m9 import frontend
from m9.nikon_matrices import lookup_nikon_matrix, nikon_matrix_catalog


def test_catalog_matches_pinned_upstream_table_and_source_files():
    catalog = nikon_matrix_catalog()
    assert catalog["schema_version"] == 1
    assert catalog["matrix_direction"] == "XYZ_to_camera"
    assert catalog["coefficient_scale"] == 10000
    assert catalog["source"]["version"] == catalog["source"]["tag"] == "0.22.1"
    assert {path: value["sha256"] for path, value in catalog["source"]["files"].items()} == {
        "src/tables/colordata.cpp": "621d6a7e1a1904095d27087cf10ac7e935c6a984ec336816b79c273a8eaca768",
        "src/metadata/normalize_model.cpp": "1e12886c679c8941eb4f84e447ba89c7c88c31f3b2a05508fd1510939eb3fdfa",
        "src/tables/cameralist.cpp": "7db5ef12c6ef5b81a81f344b798701f04beee27b340675911f848b22ec611bd7",
    }
    rows = sorted((entry["model_prefix"], entry["coefficients"]) for entry in catalog["entries"])
    # Derived independently from the pinned upstream extraction, not this catalog.
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    assert digest == "3d2a1d030eb7794f35e17b694c2e81a793d4815bb15d2c9b4d2d9f2b9ebd5085"
    assert len(rows) == 93
    assert Counter(len(coefficients) for _, coefficients in rows) == {9: 85, 12: 8}


def test_each_catalog_row_preserves_integer_coefficients_and_channel_count():
    entries = nikon_matrix_catalog()["entries"]
    for entry in entries:
        coefficients = entry["coefficients"]
        assert all(type(value) is int for value in coefficients), entry["model_prefix"]
        matrix = np.asarray(coefficients, np.float32).reshape(-1, 3) / np.float32(10000)
        assert matrix.shape in ((3, 3), (4, 3))
        assert np.isfinite(matrix).all() and np.linalg.matrix_rank(matrix) == 3
        assert type(entry["black"]) is int and type(entry["maximum"]) is int
        assert type(entry["source_line"]) is int and entry["source_line"] > 0
        if matrix.shape == (3, 3):
            frontend.camera_d50_transform(matrix)
    assert {entry["model_prefix"] for entry in entries if len(entry["coefficients"]) == 12} == {
        "COOLPIX 2500", "COOLPIX 4500", "COOLPIX 5000", "COOLPIX 5700",
        "COOLPIX 700", "COOLPIX 800", "COOLPIX 950", "COOLPIX 995",
    }


@pytest.mark.parametrize("model,coefficients", [
    ("D1", [7637, -2199, -974, -9109, 17099, 2043, -2822, 3306, 8372]),
    ("D1X", [7702, -2245, -975, -9114, 17242, 1875, -2679, 3055, 8521]),
    ("D50", [7732, -2422, -789, -8238, 15884, 2498, -859, 783, 7330]),
    ("D5000", [7309, -1403, -519, -8474, 16008, 2622, -2433, 2826, 8064]),
    ("Z6_3", [11206, -4286, -941, -4879, 12847, 2251, -745, 1654, 7374]),
    ("Z 7_2", [13705, -6004, -1400, -5464, 13568, 2062, -940, 1706, 7618]),
    ("E700", [-3746, 10611, 1665, 9621, -1734, 2114, -2389, 7082, 3064, 3406, 6116, -244]),
])
def test_fixed_anchors_keep_direction_and_specific_overlapping_models(model, coefficients):
    assert lookup_nikon_matrix("NIKON", model)["coefficients"] == coefficients


def test_every_explicit_alias_is_found_with_only_documented_normalization():
    normalized = {}
    literal_count = 0
    for entry in nikon_matrix_catalog()["entries"]:
        assert entry["models"]
        for model in entry["models"]:
            literal_count += 1
            key = " ".join(model.upper().split())
            assert normalized.setdefault(key, entry) is entry, model
            for make, name in [("NIKON", model), ("NIKON CORPORATION", "NIKON " + model),
                               ("  nikon\tcorporation ", "  nikon\t" + model.lower() + "  ")]:
                assert lookup_nikon_matrix(make, name) is entry, (make, name)
    assert literal_count == 151 and len(normalized) == 124


@pytest.mark.parametrize("make,model", [
    ("Canon", "D50"), ("NIKON CORPORATION LTD", "D50"), ("NOT NIKON", "D50"),
    ("NIKON CAMERAS", "D50"), ("NIKON_CORPORATION", "D50"), ("", "D50"),
    ("NIKON", "D50X"), ("NIKON", "D5000garbage"), ("NIKON", "D1X extra"),
    ("NIKON", "Z6_3 stale"), ("NIKON", "Z 6_3"), ("NIKON", "Z6III"),
    ("NIKON", "Z 6 II garbage"), ("NIKON", "NIKONNIKON D50"),
    ("NIKON", "NIKON NIKON D50"), ("NIKON", "UNKNOWN"), ("NIKON", ""),
    ("NIKON", "D50\x00garbage"),
])
def test_lookup_does_not_guess_maker_prefix_suffix_or_model_punctuation(make, model):
    assert lookup_nikon_matrix(make, model) is None


def test_other_upstream_transform_is_auditable_and_not_used_as_xyz_to_camera():
    extra, = nikon_matrix_catalog()["additional_upstream_matrices"]
    assert extra["matrix_direction"] == "camera_channels_to_libraw_rgb_cam"
    assert extra["matrix_shape"] == [3, 4]
    assert extra["coefficient_scale"] == 1
    assert extra["coefficients"] == [
        -1.93628, 1.800443, -1.448486, 2.584324, 1.405365, -.524955,
        -.28909, .40868, -1.204965, 1.082304, 2.941367, -1.818705,
    ]
    assert len(extra["models"]) == 6
    for model in extra["models"]:
        assert lookup_nikon_matrix("NIKON", model) is None


@pytest.mark.parametrize("model", ["COOLPIX 3700", "E3700", "COOLPIX S6", "Coolscan NEF"])
def test_models_without_a_verified_xyz_matrix_stay_unmatched(model):
    assert lookup_nikon_matrix("NIKON", model) is None
