#!/usr/bin/env python3
"""Synchronize the Project 07 development web app with the repository /docs site.

The repository publishes GitHub Pages from ``main`` and ``/docs``. The editable
frontend remains in ``07-document-semantic-search-sentence-bert/web``; this
script creates the deployable mirror at
``docs/07-document-semantic-search-sentence-bert``.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SOURCE_DIR = PROJECT_ROOT / "web"
TARGET_DIR = REPOSITORY_ROOT / "docs" / PROJECT_ROOT.name
IGNORED_NAMES = {".DS_Store", "Thumbs.db"}


def relative_files(root: Path) -> set[Path]:
    """Return all deployable files below *root* as relative paths."""
    if not root.exists():
        return set()
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name not in IGNORED_NAMES
    }


def mirror_is_current() -> tuple[bool, list[str]]:
    """Compare the source app with its /docs deployment mirror."""
    issues: list[str] = []
    source_files = relative_files(SOURCE_DIR)
    target_files = relative_files(TARGET_DIR)

    for missing in sorted(source_files - target_files):
        issues.append(f"Missing from docs mirror: {missing.as_posix()}")
    for extra in sorted(target_files - source_files):
        issues.append(f"Extra file in docs mirror: {extra.as_posix()}")
    for relative_path in sorted(source_files & target_files):
        source = SOURCE_DIR / relative_path
        target = TARGET_DIR / relative_path
        if not filecmp.cmp(source, target, shallow=False):
            issues.append(f"Different content: {relative_path.as_posix()}")

    return not issues, issues


def synchronize() -> None:
    """Replace the deployment mirror with an exact copy of the web app."""
    if not SOURCE_DIR.joinpath("index.html").is_file():
        raise FileNotFoundError(f"Expected browser app at {SOURCE_DIR / 'index.html'}")

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_DIR, TARGET_DIR)

    # Disable Jekyll processing for static model/data assets at the site root.
    nojekyll = TARGET_DIR.parent / ".nojekyll"
    nojekyll.touch(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that web/ and docs/Project-07 are identical without changing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        current, issues = mirror_is_current()
        if current:
            print(f"Deployment mirror is current: {TARGET_DIR}")
            return 0
        print("Deployment mirror is not current:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        print("Run: python scripts/sync_docs_site.py", file=sys.stderr)
        return 1

    synchronize()
    current, issues = mirror_is_current()
    if not current:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1
    print(f"Synchronized {SOURCE_DIR} -> {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
