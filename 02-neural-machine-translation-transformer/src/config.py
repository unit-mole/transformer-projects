from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def load_model_metadata(path: str | Path | None = None) -> dict[str, Any]:
    metadata_path = Path(path) if path else DEFAULT_METADATA_PATH
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata["en_hi_model_id"] = os.getenv(
        "EN_HI_MODEL_ID", metadata["en_hi_model_id"]
    )
    metadata["hi_en_model_id"] = os.getenv(
        "HI_EN_MODEL_ID", metadata["hi_en_model_id"]
    )
    return metadata
