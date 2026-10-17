"""Pinned LibRaw Nikon coefficients and exact, finite model-name lookup.

The inventory also retains four-channel and camera-to-RGB source data.  Those
records are not interchangeable with the native route's three-channel
XYZ-to-camera matrices.  Loading this module never downloads camera profiles.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


def _key(value: str) -> str:
    return " ".join(value.upper().split())


@lru_cache(maxsize=1)
def nikon_matrix_catalog() -> dict:
    return json.loads((Path(__file__).with_name("data") / "nikon_matrices.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _model_index() -> dict:
    return {_key(model): entry for entry in nikon_matrix_catalog()["entries"] for model in entry["models"]}


def lookup_nikon_matrix(make: str, model: str) -> dict | None:
    """Match only names explicitly recorded in the pinned upstream inventory."""
    if _key(make) not in ("NIKON", "NIKON CORPORATION"):
        return None
    name = _key(model)
    if name.startswith("NIKON "):
        name = name[6:]
    return _model_index().get(name)
