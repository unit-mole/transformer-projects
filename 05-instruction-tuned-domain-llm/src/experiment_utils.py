"""Reproducibility, hardware detection, run metadata, and JSON helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def save_json(data: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def utc_run_id(prefix: str = "project05") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def set_reproducibility(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def git_commit(project_dir: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "cuda_runtime": None,
        "bf16_supported": False,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
            info["compute_capability"] = list(torch.cuda.get_device_capability(0))
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            info["tf32_supported"] = bool(getattr(torch.cuda, "is_tf32_supported", lambda: False)())
    except ImportError:
        info["torch"] = None
    for module_name in ("transformers", "datasets", "peft", "accelerate", "sentence_transformers"):
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            info[module_name] = None
    return info


def choose_precision(preference: str = "auto") -> dict[str, bool | str]:
    """Choose BF16 first, then FP16, then FP32 based on the current NVIDIA GPU."""
    preference = preference.lower()
    try:
        import torch
    except ImportError:
        return {"mode": "fp32", "bf16": False, "fp16": False, "tf32": False}

    if preference not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("precision must be one of: auto, bf16, fp16, fp32")
    cuda = torch.cuda.is_available()
    bf16_supported = cuda and torch.cuda.is_bf16_supported()
    capability = torch.cuda.get_device_capability(0) if cuda else (0, 0)
    fp16_supported = cuda and capability[0] >= 7
    if preference == "bf16" and not bf16_supported:
        raise RuntimeError("BF16 was requested but is not supported by this GPU/PyTorch build.")
    if preference == "fp16" and not fp16_supported:
        raise RuntimeError("FP16 was requested but is not supported by this GPU/PyTorch build.")
    if preference == "auto":
        mode = "bf16" if bf16_supported else ("fp16" if fp16_supported else "fp32")
    else:
        mode = preference
    tf32 = bool(cuda and capability[0] >= 8)
    return {"mode": mode, "bf16": mode == "bf16", "fp16": mode == "fp16", "tf32": tf32}


def count_parameters(model: Any) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": round(100.0 * trainable / total, 6) if total else 0.0,
    }
