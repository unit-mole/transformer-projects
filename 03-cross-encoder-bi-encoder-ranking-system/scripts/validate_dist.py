from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST = PROJECT_ROOT / "web" / "dist"

REQUIRED = [
    DIST / "index.html",
    DIST / "README.md",
    DIST / "data" / "sample_documents.json",
    DIST / "data" / "sample_queries.json",
    DIST / "data" / "sample_qrels.json",
    DIST / "data" / "benchmark_summary.json",
]


def main() -> None:
    missing = [
        str(path.relative_to(DIST))
        for path in REQUIRED
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"Vite build is incomplete: {missing}")

    readme = (DIST / "README.md").read_text(encoding="utf-8")
    if "sdk: static" not in readme:
        raise SystemExit("Built Space README does not declare sdk: static.")
    if "app_file: index.html" not in readme:
        raise SystemExit(
            "Built Space README does not declare app_file: index.html."
        )

    assets = DIST / "assets"
    if not assets.exists() or not any(assets.iterdir()):
        raise SystemExit("Vite assets directory is missing or empty.")

    print("Deployable Static Space build passed validation.")
    print(f"Build directory: {DIST}")


if __name__ == "__main__":
    main()
