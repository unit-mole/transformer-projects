"""Dataset and training-result visualizations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def create_dataset_visualizations(records: Iterable[Dict[str, object]], output_dir: str | Path) -> None:
    import pandas as pd

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(records))
    if df.empty:
        return
    df["prompt_words"] = df["instruction"].fillna("").str.split().str.len() + df["input"].fillna("").str.split().str.len()
    df["response_words"] = df["output"].fillna("").str.split().str.len()
    plt = _pyplot()

    ax = df["category"].value_counts().sort_values().plot(kind="barh", figsize=(9, 6))
    ax.set_title("Instruction Category Distribution")
    ax.set_xlabel("Examples")
    ax.set_ylabel("Category")
    plt.tight_layout()
    plt.savefig(output / "instruction_category_distribution.png", dpi=160)
    plt.close()

    ax = df["prompt_words"].plot(kind="hist", bins=15, figsize=(8, 5))
    ax.set_title("Prompt Length Distribution")
    ax.set_xlabel("Words")
    plt.tight_layout()
    plt.savefig(output / "prompt_length_distribution.png", dpi=160)
    plt.close()

    ax = df["response_words"].plot(kind="hist", bins=15, figsize=(8, 5))
    ax.set_title("Response Length Distribution")
    ax.set_xlabel("Words")
    plt.tight_layout()
    plt.savefig(output / "response_length_distribution.png", dpi=160)
    plt.close()


def create_training_curve(trainer_state_path: str | Path, output_path: str | Path) -> bool:
    state_path = Path(trainer_state_path)
    if not state_path.exists():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    train = [(x.get("step"), x.get("loss")) for x in history if x.get("loss") is not None]
    evals = [(x.get("step"), x.get("eval_loss")) for x in history if x.get("eval_loss") is not None]
    if not train and not evals:
        return False
    plt = _pyplot()
    plt.figure(figsize=(8, 5))
    if train:
        plt.plot([x[0] for x in train], [x[1] for x in train], label="train loss")
    if evals:
        plt.plot([x[0] for x in evals], [x[1] for x in evals], label="validation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True
