from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from .baselines import BM25Retriever
from .beir_loader import BEIRDataset, combine_title_and_text, load_beir_dataset
from .transformer_retrieval import resolve_device


@dataclass
class FineTuneConfig:
    project_root: Path
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dataset_name: str = "scifact"
    split: str = "train"
    output_name: str = "docrank360-minilm-scifact"
    device: str = "auto"
    epochs: float = 2.0
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    hard_negative_pool: int = 50
    max_train_pairs: int | None = None
    seed: int = 42

    @property
    def output_dir(self) -> Path:
        return self.project_root / "models" / "fine_tuned" / self.output_name

    @property
    def cache_dir(self) -> Path:
        return self.project_root / "data" / "benchmarks"


def build_hard_negative_triplets(
    dataset: BEIRDataset,
    *,
    pool_size: int = 50,
    max_pairs: int | None = None,
    seed: int = 42,
) -> Dataset:
    document_ids = dataset.corpus_ids
    documents = dataset.corpus_texts
    query_ids = dataset.query_ids
    queries = dataset.query_texts

    bm25 = BM25Retriever()
    bm25.fit(documents)
    candidates = bm25.search(
        query_ids,
        queries,
        document_ids,
        top_k=pool_size,
    ).rankings

    rows: list[dict[str, str]] = []
    for query_id in query_ids:
        positives = [
            document_id
            for document_id, score in dataset.qrels[query_id].items()
            if score > 0 and document_id in dataset.corpus
        ]
        positive_set = set(positives)
        negative = next(
            (
                document_id
                for document_id in candidates[query_id]
                if document_id not in positive_set
            ),
            None,
        )
        if negative is None:
            continue
        for positive in positives:
            positive_document = dataset.corpus[positive]
            negative_document = dataset.corpus[negative]
            rows.append(
                {
                    "anchor": dataset.queries[query_id],
                    "positive": combine_title_and_text(
                        positive_document.get("title", ""),
                        positive_document.get("text", ""),
                    ),
                    "negative": combine_title_and_text(
                        negative_document.get("title", ""),
                        negative_document.get("text", ""),
                    ),
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_pairs is not None:
        rows = rows[: max(1, int(max_pairs))]
    if not rows:
        raise ValueError("No training triplets could be created.")
    return Dataset.from_list(rows)


def fine_tune_bi_encoder(config: FineTuneConfig) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    dataset = load_beir_dataset(
        config.dataset_name,
        config.cache_dir,
        split=config.split,
    )
    training_dataset = build_hard_negative_triplets(
        dataset,
        pool_size=config.hard_negative_pool,
        max_pairs=config.max_train_pairs,
        seed=config.seed,
    )

    device = resolve_device(config.device)
    model = SentenceTransformer(config.base_model, device=device)
    loss = losses.MultipleNegativesRankingLoss(model)

    use_fp16 = device.startswith("cuda")
    use_bf16 = bool(
        device.startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    if use_bf16:
        use_fp16 = False

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir / "trainer"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        run_name=config.output_name,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        loss=loss,
    )
    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.output_dir))
    metadata = {
        **asdict(config),
        "project_root": str(config.project_root),
        "output_dir": str(config.output_dir),
        "device": device,
        "training_pairs": len(training_dataset),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "training_method": "MultipleNegativesRankingLoss with BM25 hard negatives",
        "base_model": config.base_model,
    }
    (config.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata
