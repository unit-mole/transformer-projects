"""Hardware detection and safe single-GPU training presets.

The functions in this module do not require CUDA to be available. They return a
serializable report that is saved with every experiment for reproducibility.
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class HardwareProfile:
    python_version: str
    platform: str
    torch_version: str
    cuda_available: bool
    cuda_version: str
    cudnn_version: str
    gpu_name: str
    gpu_vram_gb: float
    compute_capability: str
    bf16_supported: bool
    recommended_precision: str
    recommended_model_id: str
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    dataloader_num_workers: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cpu_profile(torch_version: str = "not-installed") -> HardwareProfile:
    return HardwareProfile(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch_version,
        cuda_available=False,
        cuda_version="none",
        cudnn_version="none",
        gpu_name="none",
        gpu_vram_gb=0.0,
        compute_capability="none",
        bf16_supported=False,
        recommended_precision="fp32",
        recommended_model_id="google/flan-t5-small",
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )


def detect_hardware() -> HardwareProfile:
    """Detect CUDA capabilities and choose conservative training defaults."""
    try:
        import torch
    except ImportError:
        return _cpu_profile()

    if not torch.cuda.is_available():
        return _cpu_profile(torch.__version__)

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    vram_gb = round(props.total_memory / (1024**3), 2)
    capability_tuple = torch.cuda.get_device_capability(device_index)
    capability = f"{capability_tuple[0]}.{capability_tuple[1]}"
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    precision = "bf16" if bf16_supported else "fp16"

    # FLAN-T5-base is practical with LoRA on most modern RTX cards. The presets
    # preserve an effective batch size near 16 while reducing the micro-batch
    # on smaller cards.
    if vram_gb >= 20:
        model_id = "google/flan-t5-base"
        train_bs, eval_bs, grad_accum, checkpointing = 8, 8, 2, False
    elif vram_gb >= 12:
        model_id = "google/flan-t5-base"
        train_bs, eval_bs, grad_accum, checkpointing = 4, 4, 4, True
    elif vram_gb >= 8:
        model_id = "google/flan-t5-base"
        train_bs, eval_bs, grad_accum, checkpointing = 2, 2, 8, True
    else:
        model_id = "google/flan-t5-small"
        train_bs, eval_bs, grad_accum, checkpointing = 2, 2, 8, True

    import os

    workers = min(4, max(0, (os.cpu_count() or 2) // 2))

    return HardwareProfile(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=True,
        cuda_version=str(torch.version.cuda or "unknown"),
        cudnn_version=str(torch.backends.cudnn.version() or "unknown"),
        gpu_name=props.name,
        gpu_vram_gb=vram_gb,
        compute_capability=capability,
        bf16_supported=bf16_supported,
        recommended_precision=precision,
        recommended_model_id=model_id,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=checkpointing,
        dataloader_num_workers=workers,
    )


def save_hardware_report(path: str | Path, profile: HardwareProfile | None = None) -> Dict[str, Any]:
    """Save a reproducibility report and return it as a dictionary."""
    resolved = profile or detect_hardware()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = resolved.to_dict()
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
