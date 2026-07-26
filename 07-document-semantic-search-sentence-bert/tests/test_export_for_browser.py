import json
from pathlib import Path

from src.export_for_browser import REQUIRED_FILES, export_browser_data


def test_export_browser_data(tmp_path: Path):
    processed = tmp_path / "processed"
    web = tmp_path / "web"
    processed.mkdir()
    for filename in REQUIRED_FILES:
        (processed / filename).write_text(json.dumps({"file": filename}), encoding="utf-8")
    copied = export_browser_data(processed, web)
    assert len(copied) == len(REQUIRED_FILES)
    assert all(path.is_file() for path in copied)
