from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = PROJECT_ROOT.parent / ".github" / "workflows" / "09-vision-language-image-text-retrieval-clip.yml"

FORBIDDEN_PATTERNS = [
    re.compile(r"^\s*uses:\s*actions/(?:configure-pages|deploy-pages)@", re.IGNORECASE),
    re.compile(r"^\s*uses:\s*[^\s]*gh-pages", re.IGNORECASE),
    re.compile(r"^\s*PAGES_DEPLOY_TOKEN\s*:", re.IGNORECASE),
    re.compile(r"^\s*pages:\s*write\s*$", re.IGNORECASE),
    re.compile(r"^\s*id-token:\s*write\s*$", re.IGNORECASE),
]


def main() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in FORBIDDEN_PATTERNS):
            errors.append(f"{WORKFLOW.name}:{line_number}: {line.strip()}")

    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"Project 09 workflow is not validation-only:\n{formatted}")

    print("Project 09 workflow is validation-only and compatible with main/docs publishing.")


if __name__ == "__main__":
    main()
