"""Reproducible charts for dataset and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data_preprocessing import load_jsonl


def create_dataset_charts(dataset_path: str | Path, output_dir: str | Path) -> list[str]:
    frame = pd.DataFrame(load_jsonl(dataset_path))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame["prompt_words"] = (frame["instruction"].fillna("") + " " + frame["input"].fillna("")).str.split().str.len()
    frame["response_words"] = frame["output"].str.split().str.len()
    paths = []
    for column, title, filename in [
        ("prompt_words", "Prompt Length Distribution", "prompt_length_distribution.png"),
        ("response_words", "Response Length Distribution", "response_length_distribution.png"),
    ]:
        plt.figure(figsize=(9, 5))
        frame[column].plot(kind="hist", bins=15)
        plt.title(title)
        plt.xlabel("Words")
        plt.tight_layout()
        path = output / filename
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))
    plt.figure(figsize=(10, 6))
    frame["category"].value_counts().sort_values().plot(kind="barh")
    plt.title("Instruction Category Distribution")
    plt.xlabel("Examples")
    plt.tight_layout()
    path = output / "instruction_category_distribution.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))
    return paths
