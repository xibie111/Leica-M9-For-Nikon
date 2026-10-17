"""Single-file Nikon NEF -> tele M9 SDR JPEG, with explicit input route and no fallback."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import psutil

from .device import resolve_device
from .frontend import b2y_from_raw
from .leicafilter import LeicaFilter
from .raw_input import load_nef
from .render import _tele_zoom
from . import styletrans
from .uhdr import icc_p3, srgb_to_p3, _seg


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot record {type(value).__name__}")


class _Resources:
    """Sample process RSS; CUDA peaks come from the actual allocator, not polling."""

    def __init__(self, device):
        self.device = device
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss

    def __enter__(self):
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.05):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def finish(self):
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        result = {"peak_process_rss_bytes_sampled": self.peak_rss,
                  "rss_sample_interval_seconds": 0.05}
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()
            result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        return result

    def __exit__(self, *_):
        self.stop.set()
        self.worker.join()


@contextmanager
def _target(output_dir, stem):
    """Reserve a name without ever exposing an empty/partial formal JPEG."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index}"
        target = directory / f"{stem}{suffix}.jpg"
        lock = target.with_suffix(".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            index += 1
            continue
        os.close(fd)
        if target.exists() or target.with_suffix(".json").exists():
            lock.unlink()
            index += 1
            continue
        break
    try:
        yield target
    finally:
        lock.unlink(missing_ok=True)


def _publish(temp, final):
    # Windows rename is exclusive. POSIX rename replaces, so use exclusive hard-linking there.
    if os.name == "nt":
        os.rename(temp, final)
    else:
        os.link(temp, final)
        os.unlink(temp)


def _encode_jpeg(rgb, quality=95):
    """Convert to P3 in strips so float intermediates do not scale with full resolution."""
    pixels = np.empty(rgb.shape, np.uint8)
    for y in range(0, rgb.shape[0], 128):
        pixels[y:y + 128] = (srgb_to_p3(rgb[y:y + 128]) * 255.0 + 0.5).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", pixels[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    data = encoded.tobytes()
    return data[:2] + _seg(0xFFE2, b"ICC_PROFILE\0\x01\x01" + icc_p3()) + data[2:]


def process_nef(input_path, output_dir, method="dng", resolution="original", device="auto",
                converter=None, progress=None, write_json=False):
    """Return the processing record; optionally save it beside the JPEG with write_json=True."""
    source = Path(input_path).expanduser().resolve()
    if source.suffix.lower() != ".nef" or not source.is_file():
        raise ValueError(f"请选择存在的 NEF 文件：{source}")
    if method not in ("dng", "native") or resolution not in ("original", "phone"):
        raise ValueError("Invalid method or resolution")
    selected = resolve_device(device)
    started = time.perf_counter()
    timings = {}

    def notify(stage, message):
        if progress:
            progress(stage, message)

    @contextmanager
    def phase(name, message):
        notify(name, message)
        before = time.perf_counter()
        try:
            yield
            if selected == "cuda":
                import torch
                torch.cuda.synchronize()
        except Exception as error:
            # Fail at the selected resolution/device; a smaller or different route is never retried.
            raise RuntimeError(f"{name} 阶段失败：{error}") from error
        finally:
            timings[name] = round(time.perf_counter() - before, 4)

    with _Resources(selected) as resources:
        with phase("models", f"检查三份 StyleTrans 权重，设备 {selected}"):
            st = styletrans.load(device=selected, strict=True)
        with phase("raw", "转换并解码 RAW" if method == "dng" else "直接解码 NEF"):
            linear, cam2srgb, info = load_nef(source, method=method, converter=converter, progress=notify)
            zoom = _tele_zoom(str(source))
        source_size = [int(linear.shape[1]), int(linear.shape[0])]
        with phase("frontend", "tele M9 测光与色彩处理"):
            fe = b2y_from_raw(linear, cam2srgb, info, ev=0.0, camera="tele")
        del linear
        fe.lin = None  # SDR only: release the full-resolution linear branch before the networks.
        image = fe.rgb
        fe.rgb = None
        with phase("style", f"StyleTrans=on，{resolution} 模式"):
            image = st(image, fe.lux_index, fe.cct, resolution=resolution, progress=notify)
        with phase("filter", "LeicaFilter LUT 与暗角"):
            image = LeicaFilter(device=selected)(image, lux_index=fe.lux_index, cct=fe.cct,
                                                  zoom=zoom, scene="common", style_trans_on=True)
        model_dir = Path(st.be.wdir)
        import torch
        report = {
            "status": "complete", "input": str(source), "input_sha256": _sha256(source),
            "method": method, "resolution": resolution, "device_requested": device,
            "device": selected, "input_size": source_size,
            "output_size": [int(image.shape[1]), int(image.shape[0])],
            "camera_tuning": "tele", "zoom": zoom, "white_balance": "D50 locked", "ev": 0.0,
            "jpeg_quality": 95, "icc": "Display P3", "hdr": False,
            "raw_info": asdict(info), "cct_kelvin": float(fe.cct), "lux_index": float(fe.lux_index),
            "exposure": {**asdict(fe.ae), "short_adjustment_ev": math.log2(fe.ae.exp_short)},
            "models": {"StyleTrans": "on", "selected_style": styletrans.choose_model(fe.lux_index, fe.cct),
                       "weights_sha256": {m: _sha256(model_dir / f"{m}.npz") for m in ("low", "high", "colorfix")}},
            "software": {p: importlib.metadata.version(p) for p in
                         ("numpy", "rawpy", "tifffile", "torch", "opencv-python-headless")},
            "python": sys.version, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if selected == "cuda" else None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "timings_seconds": timings,
            "limitations": ["8-bit network; original pixel dimensions do not imply lossless RAW detail",
                            "Native NEF CCT is a single-matrix approximation; compare both routes on the same photo"],
        }
        with phase("jpeg", "编码 Display P3 JPEG"):
            data = _encode_jpeg(image)
        report["resources"] = resources.finish()
        report["timings_seconds"]["total_before_publish"] = round(time.perf_counter() - started, 4)
        # The public return value and the on-disk record must use the same JSON-compatible types.
        report = json.loads(json.dumps(report, default=_json_value, allow_nan=False))
        # Encoding is expensive; hand the already encoded data to the writer once.
        stem = f"{source.stem}_M9_tele_{method}_{resolution}"
        with _target(output_dir, stem) as target:
            report["output"] = str(target)
            report["record"] = str(target.with_suffix(".json")) if write_json else None
            _write_result(data, report, target, write_json=write_json)
    notify("done", f"完成：{report['output']}")
    return report


def _write_result(data, report, target, write_json=False):
    """Publish complete files; roll back only files published by this invocation."""
    temporary = []
    published = []
    report["jpeg_bytes"] = len(data)
    report["jpeg_sha256"] = hashlib.sha256(data).hexdigest()
    payloads = [(target, data)]
    if write_json:
        payloads.append((target.with_suffix(".json"), json.dumps(
            report, ensure_ascii=False, indent=2, default=_json_value, allow_nan=False).encode("utf-8")))
    try:
        for _, payload in payloads:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".m9-", suffix=".part", delete=False) as stream:
                temporary.append(Path(stream.name))
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for temp, (final, _) in zip(temporary, payloads):
            _publish(temp, final)
            published.append(final)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--method", choices=("dng", "native"), default="dng")
    parser.add_argument("--output-dir", type=Path, default=Path("out"))
    parser.add_argument("--resolution", choices=("original", "phone"), default="original")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--converter", type=Path, help="Adobe DNG Converter executable")
    parser.add_argument("--write-json", action="store_true", help="export a JSON processing report beside the JPEG (default: off)")
    args = parser.parse_args(argv)
    try:
        process_nef(args.input, args.output_dir, args.method, args.resolution, args.device,
                    converter=args.converter, write_json=args.write_json, progress=lambda stage, message: print(
                        f"[{stage}] {message}", flush=True))
    except Exception as error:
        print(f"处理失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
