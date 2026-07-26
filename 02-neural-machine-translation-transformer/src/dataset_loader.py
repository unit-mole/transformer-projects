from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_preprocessing import preprocess_parallel_dataframe

DEFAULT_DATASET = "cfilt/iitb-english-hindi"


def load_parallel_csv(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    return preprocess_parallel_dataframe(dataframe)


def load_iitb_dataframe(
    split: str = "validation",
    *,
    dataset_name: str = DEFAULT_DATASET,
) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required. Install requirements.txt."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    dataframe = dataset.to_pandas()
    return preprocess_parallel_dataframe(dataframe)
