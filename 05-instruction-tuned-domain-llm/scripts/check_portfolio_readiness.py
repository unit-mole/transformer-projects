#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio_readiness import assess_portfolio_readiness

result = assess_portfolio_readiness(PROJECT_ROOT)
(PROJECT_ROOT / "outputs" / "portfolio_readiness_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
