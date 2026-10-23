"""Portable-specific failures must not alter the ordinary GUI/API contract."""
import json
import os
from pathlib import Path

import pytest

from m9 import portable


@pytest.mark.parametrize("relative", ["re/weights/low.npz", "runtime/msvcp140.dll"])
def test_portable_rejects_damaged_and_missing_models_or_runtime(tmp_path, relative):
    model = tmp_path / relative
    model.parent.mkdir(parents=True)
    model.write_bytes(b"original")
    import hashlib
    entries = [{"path": relative, "bytes": 8,
                "sha256": hashlib.sha256(b"original").hexdigest()}]
    (tmp_path / "FILE_MANIFEST.json").write_text(json.dumps(entries))
    assert portable.verify_files(tmp_path) == 1
    model.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="damaged or modified"):
        portable.verify_files(tmp_path)
    model.unlink()
    with pytest.raises(FileNotFoundError):
        portable.verify_files(tmp_path)


def test_portable_does_not_inherit_external_model_or_python(monkeypatch, tmp_path):
    for key, value in {"M9_WEIGHTS": "outside", "M9_DEVICE": "cpu", "M9_BACKEND": "qnn",
                       "PYTHONPATH": "outside", "TCL_LIBRARY": "outside",
                       "M9_DNG_CONVERTER": "custom-converter"}.items():
        monkeypatch.setenv(key, value)
    # Restore every environment mutation and the process-wide bytecode flag.
    monkeypatch.setattr(portable.os, "environ", dict(os.environ))
    monkeypatch.setattr(portable.sys, "dont_write_bytecode", portable.sys.dont_write_bytecode)
    portable.configure(tmp_path)
    assert os.environ["M9_DEVICE"] == "cuda"
    assert os.environ["M9_BACKEND"] == "torch"
    assert Path(os.environ["M9_WEIGHTS"]) == tmp_path / "re" / "weights"
    assert "PYTHONPATH" not in os.environ
    assert os.environ["M9_DNG_CONVERTER"] == "custom-converter"


def test_portable_cuda_unavailable_is_failure(monkeypatch, tmp_path):
    import torch
    (tmp_path / "RELEASE.json").write_text(json.dumps({"dependencies": {}}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no available CUDA device"):
        portable.check_runtime(tmp_path)


def test_portable_window_locks_cuda_while_normal_gui_keeps_auto():
    import tkinter as tk
    from m9.gui import NefWindow
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(str(error))
    root.withdraw()
    try:
        normal = tk.Toplevel(root)
        normal.withdraw()
        assert NefWindow(normal).device.get() == "auto"
        release = tk.Toplevel(root)
        release.withdraw()
        window = portable.create_window(release)
        assert window.device.get() == "cuda"
        assert tuple(window.device_box.cget("values")) == ("cuda",)
        window.append_log("portable-log-check")
        for callback in root.tk.call("after", "info"):
            root.tk.call("after", "cancel", callback)
        for child in (normal, release):
            child.destroy()
    finally:
        root.destroy()
