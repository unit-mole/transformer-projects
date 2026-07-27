"""Pytest configuration for reliable imports in local and GitHub Actions runs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The repository contains this project in a hyphenated subfolder. Adding the
# project root explicitly guarantees that imports such as ``from src...`` work
# whether tests are started with ``pytest`` or ``python -m pytest``.
project_root_text = str(PROJECT_ROOT)
if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)
