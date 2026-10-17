"""Strict models owned by one NEF call or one serial queue run."""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import time
import traceback

from .device import resolve_device
from . import styletrans


def clear_exception_frames(error):
    """Keep diagnostic stacks while releasing tensors held by unwound frames."""
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))


class NefSession:
    def __init__(self, device="auto"):
        self._device = resolve_device(device)
        self._model = None
        self._weights = {}
        self.closed = False
        self.load_count = 0
        self.model_load_seconds = 0.0

    @property
    def device(self):
        return self._device

    @property
    def weights_sha256(self):
        return dict(self._weights)

    @property
    def loaded(self):
        return self._model is not None

    def get_model(self):
        if self.closed:
            raise RuntimeError("NEF session is closed")
        if self._model is None:
            started = time.perf_counter()
            try:
                self._model = styletrans.load(device=self.device, strict=True)
                directory = Path(self._model.be.wdir)
                for name in ("low", "high", "colorfix"):
                    digest = hashlib.sha256()
                    with (directory / f"{name}.npz").open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    self._weights[name] = digest.hexdigest()
                self.load_count += 1
            except BaseException as error:
                clear_exception_frames(error)
                try:
                    self.invalidate()
                except Exception as cleanup_error:
                    error.add_note(f"NEF model cleanup also failed: {cleanup_error}")
                raise
            finally:
                self.model_load_seconds += time.perf_counter() - started
        return self._model

    def invalidate(self):
        model, self._model = self._model, None
        self._weights = {}
        if model is not None:
            backend, model.be = model.be, None
            networks = getattr(backend, "nets", None)
            if networks is not None:
                networks.clear()
            del networks, backend, model
        gc.collect()
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()

    def close(self):
        if not self.closed:
            self.closed = True
            self.invalidate()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("NEF session is closed")
        return self

    def __exit__(self, kind, error, stack):
        if error is not None:
            clear_exception_frames(error)
        try:
            self.close()
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f"NEF session cleanup also failed: {cleanup_error}")
