r"""Explicit runtime checks. Synthetic tests are not Nikon photo/RAW acceptance.

Run from the repo: .venv\Scripts\python -m tools.validate_nef_runtime [--full]
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import psutil
import torch
from PIL import Image

from m9 import nef
from m9.frontend import RawInfo
from m9.torch_backend import TorchBackend


def synthetic_linear(width, height):
    """Gradients, fine lines/checks and colour fields, generated without full-size temporaries."""
    result = np.empty((height, width, 3), np.float32)
    x = np.arange(width, dtype=np.float32)[None, :]
    for start in range(0, height, 64):
        y = np.arange(start, min(start + 64, height), dtype=np.float32)[:, None]
        fine = ((x.astype(np.int32) + y.astype(np.int32)) % 2).astype(np.float32) * .15
        result[start:start + len(y), :, 0] = .02 + .65 * (x / width) + fine
        result[start:start + len(y), :, 1] = .02 + .5 * (y / height) + fine
        result[start:start + len(y), :, 2] = .05 + .2 * np.sin(x / 21) ** 2 + .2 * np.cos(y / 27) ** 2
    return result


def run(output_dir, full=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"evidence": "SYNTHETIC ONLY; no real NEF decoding or photographic quality acceptance",
              "real_nef_quality": "NOT_RUN", "real_nef_performance": "NOT_RUN",
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "models": [], "full_runs": []}
    for device in ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",):
        # The CPU smoke uses small style inputs; CUDA uses the production 1120x864 input.
        backend = TorchBackend.create(device=device, strict=True)
        for name in ("low", "high", "colorfix"):
            h, w = (544, 544) if name == "colorfix" else ((864, 1120) if device == "cuda" else (64, 64))
            channels = 6 if name == "colorfix" else 4
            data = np.full((1, h, w, channels), 128, np.float32)
            before = time.perf_counter()
            with nef._Resources(device) as resources:
                result = backend.run(name, data)
                assert result.shape == (1, h, w, 3) and np.isfinite(result).all()
                assert result.min() >= 0 and result.max() <= 255
                usage = resources.finish()
            item = {"device": device, "model": name, "input_shape": list(data.shape),
                    "seconds": round(time.perf_counter() - before, 4), "resources": usage,
                    "output_range": [float(result.min()), float(result.max())]}
            report["models"].append(item)
            print(json.dumps(item), flush=True)
        del backend, result, data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if full:
        source = output_dir / "SYNTHETIC_NOT_A_CAMERA_FILE.NEF"
        source.write_text("Synthetic decoder marker, not a Nikon RAW", encoding="utf-8")

        def decode(*args, **kwargs):
            return synthetic_linear(8256, 5504), np.eye(3, dtype=np.float32), RawInfo(
                exposure_time=.01, iso=100, fnumber=4, ev100=10.64, cct=5500,
                make="SYNTHETIC", model="fine lines, gradients, textures",
                extra={"synthetic": True, "real_nef_decoder": "NOT_RUN"})

        paths = {}
        for index, resolution in enumerate(("original", "original", "original", "phone")):
            with patch.object(nef, "load_nef", decode):
                item = nef.process_nef(source, output_dir, method="native", resolution=resolution,
                                       write_json=True,
                                       progress=lambda s, m: print(f"[{index + 1}:{s}] {m}", flush=True))
            paths.setdefault(resolution, item["output"])
            gc.collect()
            item["rss_after_gc_bytes"] = psutil.Process().memory_info().rss
            item["cuda_allocated_after_gc_bytes"] = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
            report["full_runs"].append(item)
            print("RUN_COMPLETE", resolution, item["timings_seconds"], item["resources"],
                  "post_gc_rss", item["rss_after_gc_bytes"], flush=True)
        # Matching locations at 100%; JPEG decode is only for inspection artifacts.
        for resolution, path in paths.items():
            with Image.open(path) as image:
                for label, x, y in (("center", 3800, 2500), ("edge", 0, 0), ("tile_seam", 496, 496)):
                    image.crop((x, y, x + 512, y + 512)).save(
                        output_dir / f"synthetic_{resolution}_{label}_100pct.png",
                        icc_profile=image.info.get("icc_profile"))
        report["continuous_three_full_original"] = "completed, synthetic input only"
    (output_dir / "runtime_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("out/nef-validation"))
    args = parser.parse_args()
    run(args.output_dir, args.full)
