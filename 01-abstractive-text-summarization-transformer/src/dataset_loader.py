from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_preprocessing import ColumnMapping, prepare_dataframe

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "xsum": {
        "path": "EdinburghNLP/xsum",
        "name": None,
        "article_column": "document",
        "summary_column": "summary",
    },
    "cnn_dailymail": {
        "path": "abisee/cnn_dailymail",
        "name": "3.0.0",
        "article_column": "article",
        "summary_column": "highlights",
    },
}


def load_csv_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return prepare_dataframe(frame)


def load_public_dataset(
    dataset_name: str = "xsum",
    *,
    split: str = "train",
    max_samples: int = 500,
    revision: str | None = None,
) -> pd.DataFrame:
    """Load a bounded public summarization dataset through Hugging Face Datasets."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose {list(DATASET_REGISTRY)}")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to load public datasets.") from exc

    config = DATASET_REGISTRY[dataset_name]
    split_expression = f"{split}[:{max_samples}]"
    kwargs: dict[str, Any] = {"split": split_expression}
    if revision:
        kwargs["revision"] = revision

    if config["name"]:
        dataset = load_dataset(config["path"], config["name"], **kwargs)
    else:
        dataset = load_dataset(config["path"], **kwargs)

    frame = dataset.to_pandas()
    return prepare_dataframe(
        frame,
        ColumnMapping(config["article_column"], config["summary_column"]),
    )
