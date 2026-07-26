from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    default_candidate_k: int = 10
    maximum_candidate_k: int = 20
    default_rerank_k: int = 5
    similarity_metric: str = "cosine"
    index_backend: str = "numpy"
    documents_path: Path = PROJECT_ROOT / "data" / "sample_documents.csv"
    queries_path: Path = PROJECT_ROOT / "data" / "sample_queries.csv"
    qrels_path: Path = PROJECT_ROOT / "data" / "sample_qrels.csv"
    index_dir: Path = PROJECT_ROOT / "models" / "vector_index"

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "Settings":
        config_path = path or PROJECT_ROOT / "config.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        models = raw.get("models", {})
        retrieval = raw.get("retrieval", {})
        data = raw.get("data", {})

        root = config_path.resolve().parent
        return cls(
            project_root=root,
            bi_encoder_model=os.getenv(
                "BI_ENCODER_MODEL",
                models.get("bi_encoder", cls.bi_encoder_model),
            ),
            cross_encoder_model=os.getenv(
                "CROSS_ENCODER_MODEL",
                models.get("cross_encoder", cls.cross_encoder_model),
            ),
            device=os.getenv("MODEL_DEVICE", models.get("device", "cpu")),
            default_candidate_k=int(
                retrieval.get("default_candidate_k", cls.default_candidate_k)
            ),
            maximum_candidate_k=int(
                retrieval.get("maximum_candidate_k", cls.maximum_candidate_k)
            ),
            default_rerank_k=int(
                retrieval.get("default_rerank_k", cls.default_rerank_k)
            ),
            similarity_metric=str(
                retrieval.get("similarity_metric", cls.similarity_metric)
            ),
            index_backend=str(retrieval.get("index_backend", cls.index_backend)),
            documents_path=root / data.get("documents_path", "data/sample_documents.csv"),
            queries_path=root / data.get("queries_path", "data/sample_queries.csv"),
            qrels_path=root / data.get("qrels_path", "data/sample_qrels.csv"),
            index_dir=root / "models" / "vector_index",
        )
