"""Regenerate the pinned Nikon coefficient inventory; no runtime downloads.

Usage: python tools/update_nikon_matrices.py --source-dir /path/to/LibRaw-0.22.1
The source files must match the pinned SHA-256 values below.  ``--check`` checks
the committed output without writing it.  Only explicitly listed camera models
are expanded from LibRaw's prefixes; unknown models are never inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


VERSION = "0.22.1"
SOURCE_HASHES = {
    "src/tables/colordata.cpp": "621d6a7e1a1904095d27087cf10ac7e935c6a984ec336816b79c273a8eaca768",
    "src/metadata/normalize_model.cpp": "1e12886c679c8941eb4f84e447ba89c7c88c31f3b2a05508fd1510939eb3fdfa",
    "src/tables/cameralist.cpp": "7db5ef12c6ef5b81a81f344b798701f04beee27b340675911f848b22ec611bd7",
}
OUTPUT = Path(__file__).resolve().parents[1] / "m9/data/nikon_matrices.json"


def _without_comments(text: str) -> str:
    # Keep line numbers, but exclude the obsolete commented-out D1 matrix.
    return re.sub(r"/\*.*?\*/|//[^\n]*", lambda m: "\n" * m[0].count("\n"), text, flags=re.S)


def _key(model: str) -> str:
    return " ".join(model.upper().split())


def generate(source_dir: Path) -> dict:
    sources = {}
    manifest = {}
    for relative, expected in SOURCE_HASHES.items():
        data = (source_dir / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"Source hash mismatch for {relative}: {digest}")
        sources[relative] = data.decode("utf-8")
        manifest[relative] = {
            "url": f"https://raw.githubusercontent.com/LibRaw/LibRaw/{VERSION}/{relative}",
            "sha256": digest,
        }

    table = _without_comments(sources["src/tables/colordata.cpp"])
    pattern = re.compile(
        r'\{\s*LIBRAW_CAMERAMAKER_Nikon\s*,\s*"([^"]+)"\s*,'
        r"\s*(-?\w+)\s*,\s*(\w+)\s*,\s*\{([^}]+)\}\s*\}"
    )
    entries = []
    for match in pattern.finditer(table):
        coefficients = [int(value.strip(), 0) for value in match[4].split(",") if value.strip()]
        if len(coefficients) not in (9, 12):
            raise ValueError(f"Unexpected coefficient count for {match[1]}")
        entries.append({
            "model_prefix": match[1],
            "black": int(match[2], 0),
            "maximum": int(match[3], 0),
            "coefficients": coefficients,
            "models": [],
            "source_line": table.count("\n", 0, match.start()) + 1,
        })
    if len(entries) != table.count("LIBRAW_CAMERAMAKER_Nikon"):
        raise ValueError("Not every Nikon table entry was parsed")

    normalization = _without_comments(sources["src/metadata/normalize_model.cpp"])
    alias_block = re.search(r"nikonalias\[\]\[16\]\s*=\s*\{(.*?)\};", normalization, re.S)
    if alias_block is None:
        raise ValueError("Missing upstream Nikon aliases")
    aliases = {}
    canonical = None
    for value in re.findall(r'"([^"]+)"', alias_block[1]):
        if value.startswith("@"):
            canonical = value[1:]
        elif canonical:
            aliases[value] = canonical

    camera_list = _without_comments(sources["src/tables/cameralist.cpp"])
    inventory = []
    for encoded in re.findall(r'"(Nikon (?:[^"\\]|\\.)*)"', camera_list):
        name = json.loads('"' + encoded + '"')[6:]
        # Parentheses in this inventory describe file-format limitations.
        name = re.sub(r"\s*\(.*\)$", "", name).strip()
        if name not in inventory:
            inventory.append(name)

    # cameralist uses the retail generation name; EXIF uses these spellings.
    # The II spellings are documented beside the Z 6 / Z 7 coefficient rows.
    explicit_aliases = {"Z 6 III": "Z6_3", "Z 6_2": "Z 6 II", "Z 7_2": "Z 7 II"}
    models = list(dict.fromkeys(inventory + list(aliases) + list(aliases.values()) + list(explicit_aliases)))
    for entry in entries:
        if entry["model_prefix"].strip() != "1":
            models.append(entry["model_prefix"])
    models = list(dict.fromkeys(models))

    def resolve(model: str) -> dict | None:
        name = explicit_aliases.get(model, aliases.get(model, model))
        # This prefix operation is only used on the finite upstream inventory,
        # never on a photograph's model string at runtime.
        for entry in entries:
            if _key(name).startswith(_key(entry["model_prefix"])):
                return entry
        return None

    unmatched = []
    for model in models:
        entry = resolve(model)
        if entry is None:
            unmatched.append(model)
        elif model not in entry["models"]:
            entry["models"].append(model)
    for entry in entries:
        entry["models"].sort(key=_key)
        if not entry["models"]:
            raise ValueError(f"No explicit model for {entry['model_prefix']!r}")
    # An exact-model runtime lookup must not have a conflicting alias.
    seen = {}
    for entry in entries:
        for model in entry["models"]:
            key = _key(model)
            if key in seen and seen[key] != entry["model_prefix"]:
                raise ValueError(f"Conflicting model alias: {model}")
            seen[key] = entry["model_prefix"]

    simple_start = table.index("void LibRaw::simple_coeff(int index)")
    simple_table = re.search(r"table\[\]\[12\]\s*=\s*\{(.*?)\};", table[simple_start:], re.S)
    if simple_table is None:
        raise ValueError("Missing upstream simple_coeff table")
    simple_rows = re.findall(r"\{([^{}]+)\}", simple_table[1])
    if len(simple_rows) != 4:
        raise ValueError("Unexpected upstream simple_coeff table")
    simple_coefficients = [float(value.strip().removesuffix("f")) for value in simple_rows[3].split(",")]
    if len(simple_coefficients) != 12:
        raise ValueError("Unexpected Nikon simple_coeff dimension")
    additional_models = ["COOLPIX 880", "E880", "COOLPIX 900", "E900", "COOLPIX 990", "E990"]
    without_adobe = sorted(set(inventory) & set(unmatched), key=_key)
    additional_keys = {_key(model) for model in additional_models}

    return {
        "schema_version": 1,
        "source": {
            "project": "LibRaw",
            "version": VERSION,
            "tag": VERSION,
            "release_url": f"https://github.com/LibRaw/LibRaw/releases/tag/{VERSION}",
            "files": manifest,
            "recovered_from": {
                "archive": "rawpy-0.27.1.tar.gz",
                "url": "https://pypi.org/project/rawpy/0.27.1/#files",
                "sha256": "3194d64ff690ac945e1a43237edae8a18f1f493751924de1ae2bcef473c0fb79",
                "path": "rawpy-0.27.1/external/LibRaw",
            },
            "license": "CDDL-1.0",
            "notice": "LibRaw-NOTICE.txt",
        },
        "matrix_direction": "XYZ_to_camera",
        "coefficient_scale": 10000,
        "black_and_maximum_policy": "Upstream metadata retained for audit only; never applied by this inventory.",
        "entries": entries,
        "additional_upstream_matrices": [{
            "source_function": "LibRaw::simple_coeff",
            "source_index": 3,
            "source_file": "src/tables/colordata.cpp",
            "source_line": table.count("\n", 0, simple_start) + 1,
            "models": additional_models,
            "matrix_direction": "camera_channels_to_libraw_rgb_cam",
            "matrix_shape": [3, 4],
            "coefficient_scale": 1,
            "source_coefficient_type": "C++ float",
            "coefficients": simple_coefficients,
            "runtime_policy": "Audit only; never used as an XYZ-to-camera matrix or by native NEF fallback.",
        }],
        "coverage": {
            "matrix_entries": len(entries),
            "three_channel_entries": sum(len(e["coefficients"]) == 9 for e in entries),
            "four_channel_entries": sum(len(e["coefficients"]) == 12 for e in entries),
            "camera_inventory_models": sorted(inventory, key=_key),
            "inventory_without_adobe_coefficients": without_adobe,
            "inventory_without_either_matrix_table": [model for model in without_adobe if _key(model) not in additional_keys],
            "aliases_without_matrix": sorted(set(unmatched) - set(inventory), key=_key),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = generate(args.source_dir)
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != serialized:
            raise SystemExit(f"Out-of-date Nikon inventory: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(f"{'Verified' if args.check else 'Generated'} {len(result['entries'])} Nikon matrix entries")


if __name__ == "__main__":
    main()
