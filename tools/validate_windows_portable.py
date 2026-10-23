"""Real-photo portable/source comparison; all evidence stays outside the ZIP.

Use `matrix --baseline ... --bundle ... --output ...` from the development
environment. Requires the owner's Z6 III/Z7 II fixtures and Adobe converter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def arm(root, config, output):
    sys.path.insert(0, str(root))
    from m9 import nef
    assert Path(nef.__file__).resolve().is_relative_to(root)
    cases = json.loads(config.read_text(encoding="utf-8"))
    results = []
    output.mkdir()
    os.environ["M9_WEIGHTS"] = str(root / "re" / "weights")
    os.environ["M9_BACKEND"] = "torch"
    for case in cases:
        source = Path(case["input"])
        before = sha(source)
        started = time.perf_counter()
        try:
            report = nef.process_nef(source, output / case["id"], method=case["method"],
                                     resolution=case["resolution"], device="cuda")
        except Exception as error:
            traceback.print_exc()
            if before != sha(source):
                raise RuntimeError("Source photo changed") from error
            results.append({"id": case["id"], "input_sha256": before,
                            "error": str(error), "status": "FAIL"})
            save(output / "results.json", results)
            continue
        elapsed = time.perf_counter() - started
        if before != sha(source):
            raise RuntimeError("Source photo changed")
        results.append({"id": case["id"], "input_sha256": before, "seconds": elapsed,
                        "report": report, "jpeg_sha256": sha(Path(report["output"]))})
        save(output / "results.json", results)
        print(f"PASS {case['id']} {elapsed:.2f}s", flush=True)


def matrix(baseline, bundle, output, z6, z7):
    output.mkdir()
    cases = []
    for model, landscape, portrait in (("z6iii", z6 / "DSC_0067.NEF", z6 / "DSC_0015.NEF"),
                                       ("z7ii", z7 / "DSC_0147.NEF", z7 / "DSC_0237.NEF")):
        for method, resolution, photo in (("native", "original", landscape), ("native", "phone", portrait),
                                           ("dng", "original", portrait), ("dng", "phone", landscape)):
            cases.append({"id": f"{model}_{method}_{resolution}", "input": str(photo),
                          "method": method, "resolution": resolution})
    config = output / "cases.json"
    save(config, cases)
    script = str(Path(__file__).resolve())
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    for label, interpreter, source in (("source", sys.executable, baseline),
                                        ("portable", bundle / "runtime" / "python.exe", bundle)):
        with (output / (label + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run([str(interpreter), "-B", "-u", script, "arm", "--root", str(source),
                            "--config", str(config), "--output", str(output / label)],
                           env=environment, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    compare(output / "source", output / "portable", output / "comparison.json")


def compare(source, portable, output):
    from PIL import Image
    reference = json.loads((source / "results.json").read_text(encoding="utf-8"))
    candidate = json.loads((portable / "results.json").read_text(encoding="utf-8"))
    comparisons = []
    for left, right in zip(reference, candidate, strict=True):
        assert left["id"] == right["id"]
        if "error" in left or "error" in right:
            converter_blocked = "_dng_" in left["id"] and all(
                "Cannot init cloud download cache" in item.get("error", "") for item in (left, right))
            comparisons.append({"id": left["id"], "checks": {"rendered": False},
                                "errors_match": left.get("error") == right.get("error"),
                                "same_converter_failure": converter_blocked,
                                "source_error": left.get("error"), "portable_error": right.get("error")})
            continue
        with Image.open(left["report"]["output"]) as a, Image.open(right["report"]["output"]) as b:
            checks = {"jpeg_bytes": left["jpeg_sha256"] == right["jpeg_sha256"],
                      "source_hash": left["input_sha256"] == right["input_sha256"],
                      "dimensions": a.size == b.size,
                      "pixels": a.tobytes() == b.tobytes(),
                      "icc": a.info.get("icc_profile") == b.info.get("icc_profile"),
                      "exif": a.info.get("exif") == b.info.get("exif")}
            comparisons.append({"id": left["id"], "checks": checks, "dimensions": list(a.size),
                                "source_seconds": left["seconds"], "portable_seconds": right["seconds"]})
    blocked = [item for item in comparisons if item.get("same_converter_failure")]
    failed = [item for item in comparisons if not item.get("same_converter_failure")
              and not all(item["checks"].values())]
    status = "FAIL" if failed else "PARTIAL" if blocked else "PASS"
    summary = {"status": status, "matched_pairs": len(comparisons) - len(blocked) - len(failed),
               "blocked_pairs": len(blocked), "failed_pairs": len(failed), "pairs": comparisons}
    save(output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "pairs"}), flush=True)
    if failed:
        raise RuntimeError("Portable photo differs from frozen source")
    if blocked:
        raise SystemExit(2)


def gui(bundle, output, z6, z7):
    """Drive real Tk callbacks and real photos; this is not a visual review."""
    sys.path.insert(0, str(bundle))
    from m9 import portable
    assert Path(portable.__file__).resolve().is_relative_to(bundle)
    portable.configure(bundle)
    import tkinter as tk
    import psutil
    import torch
    import shutil

    output.mkdir()
    photos = [z6 / "DSC_0067.NEF", z7 / "DSC_0237.NEF"]
    before = {str(path): sha(path) for path in photos}
    damaged = output / "损坏样片.NEF"
    damaged.write_bytes(b"Deliberately invalid NEF for error reporting")
    scenarios = []

    def pump(window, until, action=None):
        deadline = time.monotonic() + 300
        requested = False
        while not until():
            if time.monotonic() > deadline:
                window.request_stop()
                raise TimeoutError("GUI scenario exceeded five minutes")
            window.root.update()
            if action and not requested and window.current_stage.get().startswith("当前阶段：raw"):
                action()
                requested = True
            time.sleep(.01)
        return requested

    for mode in ("queue_and_retry", "stop", "close"):
        root = tk.Tk()
        root.withdraw()
        window = portable.create_window(root)
        destination = output / mode
        window.output_dir.set(str(destination))
        window.add_paths([damaged, *photos] if mode == "queue_and_retry" else photos)
        try:
            window.start("native")
            action = window.request_stop if mode == "stop" else window.confirm_close_after_current if mode == "close" else None
            requested = pump(window, lambda: window.closed if mode == "close" else not window.busy, action)
            first_summary = dict(window.last_summary)
            receipt = window.job._plan_receipt
            assert not window.job.running and not window.job._cleanup_unsafe
            assert receipt["cleanup_safe"] is True
            if mode == "queue_and_retry":
                assert first_summary["succeeded"] == 2 and first_summary["failed"] == 1
                # Retry a formerly damaged input, preserving the external sources.
                shutil.copyfile(photos[0], damaged)
                window.retry_failed()
                window.start("native")
                pump(window, lambda: not window.busy)
                assert window.last_summary["succeeded"] == 1 and window.last_summary["failed"] == 0
                retry_summary = dict(window.last_summary)
                # Requeue a successful photo to exercise real no-overwrite output.
                old = window.photo_queue.items[-1]
                old_output = Path(old.report["output"])
                old_hash = sha(old_output)
                old.status = "pending"
                window.start("native")
                pump(window, lambda: not window.busy)
                assert Path(old.report["output"]) != old_output and sha(old_output) == old_hash
                scenarios.append({"mode": mode, "status": "PASS", "first_summary": first_summary,
                                  "retry_summary": retry_summary, "no_overwrite": True, "receipt": receipt})
            else:
                assert requested and first_summary["succeeded"] == 1 and first_summary["stopped"] is True
                assert len(window.photo_queue.pending_items) == 1
                scenarios.append({"mode": mode, "status": "PASS", "summary": first_summary, "receipt": receipt})
            assert torch.cuda.memory_allocated() == 0
        finally:
            if window.job.running:
                window.request_stop()
                window.job.thread.join(300)
                if window.job.running:
                    raise TimeoutError("GUI worker did not stop")
            if not window.closed:
                window._destroy_window()
        save(output / "gui-results.json", scenarios)
    assert all(sha(Path(path)) == value for path, value in before.items())
    remaining = []
    for process in psutil.Process().children(recursive=True):
        try:
            if "m9.native_decode_worker" in process.cmdline():
                remaining.append(process.pid)
        except psutil.NoSuchProcess:
            pass
    assert not remaining, remaining
    save(output / "gui-results.json", {"status": "PASS", "scenarios": scenarios,
                                      "sources_unchanged": True, "remaining_helpers": remaining,
                                      "visual_review": "NOT_RUN"})
    print("Real Tk queue/stop/close/retry/no-overwrite: PASS", flush=True)


def main():
    for console in (sys.stdout, sys.stderr):
        if console is not None and hasattr(console, "reconfigure"):
            console.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("matrix")
    for name in ("baseline", "bundle", "output"):
        run.add_argument("--" + name, required=True, type=Path)
    run.add_argument("--z6", type=Path, default=Path("E:/街拍"))
    run.add_argument("--z7", type=Path, default=Path("E:/m9input"))
    single = commands.add_parser("arm")
    for name in ("root", "config", "output"):
        single.add_argument("--" + name, required=True, type=Path)
    window = commands.add_parser("gui")
    window.add_argument("--bundle", required=True, type=Path)
    window.add_argument("--output", required=True, type=Path)
    window.add_argument("--z6", type=Path, default=Path("E:/街拍"))
    window.add_argument("--z7", type=Path, default=Path("E:/m9input"))
    pairs = commands.add_parser("compare")
    for name in ("source", "portable", "output"):
        pairs.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.command == "matrix":
        matrix(args.baseline.resolve(), args.bundle.resolve(), args.output.resolve(), args.z6.resolve(), args.z7.resolve())
    elif args.command == "arm":
        arm(args.root.resolve(), args.config.resolve(), args.output.resolve())
    elif args.command == "gui":
        gui(args.bundle.resolve(), args.output.resolve(), args.z6.resolve(), args.z7.resolve())
    else:
        compare(args.source.resolve(), args.portable.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
