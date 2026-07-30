from __future__ import annotations

from dataclasses import dataclass
import base64
import json
from pathlib import Path, PurePosixPath
from typing import Iterable

import requests

ALLOWED_FILE_NAMES = {
    "README.md",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "README_HUGGINGFACE.md",
    "README_GITHUB_PAGES.md",
    "README_VERCEL.md",
    "README_CLOUDFLARE.md",
    "PROJECT_ROADMAP.md",
}

EXCLUDED_PARTS = {
    ".git",
    "node_modules",
    ".next",
    "venv",
    ".venv",
    "data/raw",
    "private",
    "confidential",
}


@dataclass(frozen=True)
class RepositoryConfig:
    owner: str
    repo: str
    branch: str = "main"
    category: str = "Portfolio"


class GitHubPortfolioCollector:
    def __init__(self, token: str | None = None, timeout: int = 30) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "ai-portfolio-rag-assistant",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _get_json(self, url: str) -> dict | list:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def list_markdown_files(self, config: RepositoryConfig) -> list[dict]:
        url = (
            f"https://api.github.com/repos/{config.owner}/{config.repo}/git/trees/"
            f"{config.branch}?recursive=1"
        )
        payload = self._get_json(url)
        if not isinstance(payload, dict) or "tree" not in payload:
            raise RuntimeError(f"Unexpected GitHub tree response for {config.owner}/{config.repo}")

        results = []
        for entry in payload["tree"]:
            if entry.get("type") != "blob":
                continue
            path = str(entry.get("path", ""))
            pure_path = PurePosixPath(path)
            lowered = path.lower()
            if any(part in lowered for part in EXCLUDED_PARTS):
                continue
            if pure_path.name in ALLOWED_FILE_NAMES or pure_path.name.startswith("README"):
                results.append(entry)
        return results

    def fetch_text(self, config: RepositoryConfig, path: str) -> str:
        url = f"https://api.github.com/repos/{config.owner}/{config.repo}/contents/{path}?ref={config.branch}"
        payload = self._get_json(url)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected GitHub content response for {path}")
        encoding = payload.get("encoding")
        content = payload.get("content")
        if encoding != "base64" or not isinstance(content, str):
            raise RuntimeError(f"Unsupported content encoding for {path}")
        return base64.b64decode(content).decode("utf-8", errors="replace")

    def collect(self, config: RepositoryConfig, output_root: Path) -> list[Path]:
        written: list[Path] = []
        for entry in self.list_markdown_files(config):
            source_path = str(entry["path"])
            text = self.fetch_text(config, source_path)
            destination = output_root / config.category / config.repo / source_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            written.append(destination)
        return written


def load_repository_configs(path: Path) -> list[RepositoryConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("repositories", payload)
    if not isinstance(records, list):
        raise ValueError("Repository config must contain a list")
    return [RepositoryConfig(**record) for record in records]


def collect_repositories(
    configs: Iterable[RepositoryConfig],
    output_root: Path,
    token: str | None = None,
) -> list[Path]:
    collector = GitHubPortfolioCollector(token=token)
    all_written: list[Path] = []
    for config in configs:
        all_written.extend(collector.collect(config, output_root))
    return all_written
