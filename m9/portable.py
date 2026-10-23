"""Windows portable entry point; the normal GUI/CLI defaults stay unchanged."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        if self.console is not None:
            self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        if self.console is not None:
            self.console.flush()
        self.log.flush()


def configure(root=ROOT):
    """Use only bundled models/Tcl; do not inherit developer overrides."""
    for name in list(os.environ):
        if name.startswith("M9_") and name != "M9_DNG_CONVERTER":
            os.environ.pop(name)
    for name in ("PYTHONHOME", "PYTHONPATH", "TCLLIBPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        os.environ.pop(name, None)
    os.environ.update(M9_DEVICE="cuda", M9_BACKEND="torch",
                      M9_WEIGHTS=str(root / "re" / "weights"),
                      TCL_LIBRARY=str(root / "runtime" / "tcl" / "tcl8.6"),
                      TK_LIBRARY=str(root / "runtime" / "tcl" / "tk8.6"),
                      PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True


def verify_files(root=ROOT, *, full=False):
    manifest = json.loads((root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for entry in manifest:
        relative = Path(entry["path"])
        if not full and relative.parts[0] not in ("m9", "re"):
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Invalid file manifest path")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                digest.update(block)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Bundled file is damaged or modified: {relative}")
        checked += 1
    print(f"File integrity: PASS ({checked} files)", flush=True)
    return checked


def check_runtime(root=ROOT):
    import torch
    from .device import resolve_device

    release = json.loads((root / "RELEASE.json").read_text(encoding="utf-8"))
    if sys.version_info[:3] != (3, 12, 4):
        raise RuntimeError("This candidate requires bundled Python 3.12.4")
    for name, version in release["dependencies"].items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise RuntimeError(f"Dependency mismatch: {name} {actual}, expected {version}")
    resolve_device("cuda")
    tensor = torch.arange(1024, dtype=torch.float32, device="cuda")
    value = (tensor * tensor).sum().item()
    torch.cuda.synchronize()
    if abs(value - 357389824.0) > 128.0:
        raise RuntimeError(f"CUDA tensor self-check failed: {value}")
    print(json.dumps({"version": release["version"], "python": sys.version.split()[0],
                      "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(),
                      "vram_bytes": torch.cuda.get_device_properties(0).total_memory},
                     ensure_ascii=False, indent=2), flush=True)
    print("CUDA self-check: PASS. Real photo rendering is a separate check.", flush=True)


def create_window(root):
    from .gui import NefWindow

    class PortableWindow(NefWindow):
        def append_log(self, message):
            super().append_log(message)
            print(message, flush=True)

    window = PortableWindow(root)
    window.device.set("cuda")
    window.device_box.configure(values=("cuda",))
    root.title("Leica M9 For Nikon · NVIDIA 候选版")
    return window


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true", help="verify all files, Tk and all three networks")
    args = parser.parse_args(argv)
    for console in (sys.stdout, sys.stderr):
        if console is not None and hasattr(console, "reconfigure"):
            console.reconfigure(encoding="utf-8", errors="backslashreplace")
    log_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "LeicaM9ForNikon" / "logs"
    try:
        log_root.mkdir(parents=True, exist_ok=True)
        path = log_root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}.log")
        log = path.open("x", encoding="utf-8")
    except OSError:
        traceback.print_exc()
        return 1
    with log, redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
        tk_root = None
        try:
            print(f"Log: {path}", flush=True)
            configure()
            verify_files(full=args.diagnose)
            check_runtime()
            import tkinter as tk
            tk_root = tk.Tk()
            if args.diagnose:
                tk_root.withdraw()
                from .session import NefSession
                with NefSession("cuda") as session:
                    model = session.get_model()
                    import numpy as np
                    for name in ("low", "high", "colorfix"):
                        side = 544 if name == "colorfix" else 64
                        channels = 6 if name == "colorfix" else 4
                        result = model.be.run(name, np.full((1, side, side, channels), 128, np.float32))
                        if result.shape != (1, side, side, 3) or not np.isfinite(result).all():
                            raise RuntimeError(f"Network self-check failed: {name}")
                        print(f"Network {name}: PASS", flush=True)
                print("Diagnostics: PASS. No camera photo was rendered.", flush=True)
            else:
                window = create_window(tk_root)
                tk_root.mainloop()
                del window
            return 0
        except Exception:
            traceback.print_exc()
            print("启动或诊断失败。请查看以上错误及日志；不会自动改用 CPU、降低分辨率或更换模型。", flush=True)
            return 1
        finally:
            if tk_root is not None:
                try:
                    tk_root.destroy()
                except tk.TclError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
