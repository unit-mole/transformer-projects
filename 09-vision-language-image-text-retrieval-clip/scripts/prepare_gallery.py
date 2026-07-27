from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import validate_gallery_assets, write_validation_report


def main() -> None:
    web_root = PROJECT_ROOT / "web"
    report = validate_gallery_assets(web_root / "data" / "image_gallery.json", web_root)
    write_validation_report(report, PROJECT_ROOT / "outputs" / "gallery_validation.json")
    print(f"Validated {report['images']} images across {report['categories']} categories.")


if __name__ == "__main__":
    main()
