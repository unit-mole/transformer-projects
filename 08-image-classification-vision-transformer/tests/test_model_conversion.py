from pathlib import Path
import pytest
from src.model_conversion import validate_export_directory


def test_incomplete_export_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        validate_export_directory(tmp_path)
