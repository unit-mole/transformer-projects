from __future__ import annotations

import re

FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def clean_markdown(text: str) -> str:
    """Conservatively clean Markdown while preserving technical terms and headings."""
    text = FRONTMATTER.sub("", text)
    text = HTML_COMMENT.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_headings(text: str) -> list[str]:
    return [match.group(2).strip() for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE)]
