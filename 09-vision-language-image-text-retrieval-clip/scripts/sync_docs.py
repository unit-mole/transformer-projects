from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
WEB_ROOT = PROJECT_ROOT / "web"
DOCS_ROOT = REPOSITORY_ROOT / "docs" / PROJECT_ROOT.name

IGNORED_NAMES = {".DS_Store", "Thumbs.db", "__pycache__"}


def iter_relative_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name not in IGNORED_NAMES
    }


def compare_directories() -> list[str]:
    if not DOCS_ROOT.exists():
        return [f"Deployment folder is missing: {DOCS_ROOT}"]

    source_files = iter_relative_files(WEB_ROOT)
    deployed_files = iter_relative_files(DOCS_ROOT)
    errors: list[str] = []

    for relative_path in sorted(source_files - deployed_files):
        errors.append(f"Missing from docs copy: {relative_path.as_posix()}")
    for relative_path in sorted(deployed_files - source_files):
        errors.append(f"Extra file in docs copy: {relative_path.as_posix()}")

    for relative_path in sorted(source_files & deployed_files):
        source = WEB_ROOT / relative_path
        deployed = DOCS_ROOT / relative_path
        if not filecmp.cmp(source, deployed, shallow=False):
            errors.append(f"Out-of-sync file: {relative_path.as_posix()}")

    return errors


def sync() -> None:
    if not WEB_ROOT.exists():
        raise FileNotFoundError(f"Development web folder not found: {WEB_ROOT}")

    if DOCS_ROOT.exists():
        shutil.rmtree(DOCS_ROOT)
    DOCS_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(WEB_ROOT, DOCS_ROOT)
    (DOCS_ROOT.parent / ".nojekyll").touch(exist_ok=True)
    print(f"Synchronized {WEB_ROOT} -> {DOCS_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the Project 09 web app into the repository's main/docs publishing folder."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when web/ and docs/09-.../ differ instead of copying files.",
    )
    args = parser.parse_args()

    if args.check:
        errors = compare_directories()
        if errors:
            formatted = "\n".join(f"- {error}" for error in errors)
            raise SystemExit(f"Project 09 deployment copy is not synchronized:\n{formatted}")
        print("Project 09 web/ and docs deployment copies are synchronized.")
        return

    sync()


if __name__ == "__main__":
    main()
