from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def export_clip_onnx(model_id: str, output_dir: str | Path) -> Path:
    if shutil.which("optimum-cli") is None:
        raise RuntimeError("optimum-cli is not installed; install requirements-model.txt")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    command = ["optimum-cli", "export", "onnx", "--model", model_id, "--task", "feature-extraction", str(target)]
    subprocess.run(command, check=True)
    return target
