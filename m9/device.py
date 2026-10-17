"""One explicit device choice shared by the networks and LeicaFilter."""
from __future__ import annotations

import os


def resolve_device(requested: str | None = "auto") -> str:
    requested = requested or os.environ.get("M9_DEVICE", "auto")
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError("device must be auto, cuda or cpu")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as error:
        if requested == "cuda":
            raise RuntimeError("CUDA requested but PyTorch is not installed") from error
        return "cpu"
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but this PyTorch installation has no available CUDA device")
    return "cuda" if available else "cpu"
