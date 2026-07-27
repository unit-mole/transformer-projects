from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"

TEXT_EXTENSIONS = {".html", ".css", ".js", ".json"}

# Reject local references beginning at the domain root. Protocol-relative URLs (//...),
# data URLs, fragment links, and normal external URLs are not matched.
PATTERNS = [
    re.compile(r"(?:src|href)\s*=\s*[\"']/(?!/)[^\"']+[\"']", re.IGNORECASE),
    re.compile(r"fetch\(\s*[\"']/(?!/)[^\"']+[\"']", re.IGNORECASE),
    re.compile(r"url\(\s*[\"']?/(?!/)[^)\"']+", re.IGNORECASE),
    re.compile(r"from\s+[\"']/(?!/)[^\"']+[\"']", re.IGNORECASE),
]


def main() -> None:
    errors: list[str] = []
    for path in WEB_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in PATTERNS:
                if pattern.search(line):
                    relative = path.relative_to(PROJECT_ROOT).as_posix()
                    errors.append(f"{relative}:{line_number}: {line.strip()}")

    if errors:
        formatted = "\n".join(f"- {item}" for item in errors)
        raise SystemExit(
            "Repository-root asset paths were found. Project 09 is hosted in a docs subfolder; "
            f"use ./ relative paths instead:\n{formatted}"
        )

    print("All Project 09 local web references use deployment-safe relative paths.")


if __name__ == "__main__":
    main()
