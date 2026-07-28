from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    train_examples: int | None
    validation_examples: int | None
    max_length: int
    stride: int
    epochs: float
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    save_steps: int
    logging_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, TrainingProfile] = {
    "smoke": TrainingProfile(
        name="smoke",
        train_examples=32,
        validation_examples=16,
        max_length=768,
        stride=128,
        epochs=1.0,
        learning_rate=2e-5,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=2,
        save_steps=20,
        logging_steps=5,
    ),
    "portfolio": TrainingProfile(
        name="portfolio",
        train_examples=800,
        validation_examples=120,
        max_length=1536,
        stride=256,
        epochs=2.0,
        learning_rate=1.5e-5,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=100,
        logging_steps=20,
    ),
    "full": TrainingProfile(
        name="full",
        train_examples=None,
        validation_examples=None,
        max_length=2048,
        stride=256,
        epochs=3.0,
        learning_rate=1e-5,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=200,
        logging_steps=25,
    ),
    "high-vram": TrainingProfile(
        name="high-vram",
        train_examples=None,
        validation_examples=None,
        max_length=3072,
        stride=384,
        epochs=2.0,
        learning_rate=1e-5,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=200,
        logging_steps=25,
    ),
}


def recommend_profile() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "recommended_profile": "smoke",
                "reason": "CUDA is unavailable; use smoke only or install CUDA-enabled PyTorch.",
                "cuda_available": False,
            }
        properties = torch.cuda.get_device_properties(0)
        vram_gb = properties.total_memory / 1024**3
        if vram_gb >= 20:
            profile = "high-vram"
        elif vram_gb >= 10:
            profile = "portfolio"
        else:
            profile = "portfolio"
        return {
            "recommended_profile": profile,
            "reason": f"Detected {vram_gb:.1f} GB GPU memory.",
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": vram_gb,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    except Exception as exc:
        return {
            "recommended_profile": "smoke",
            "reason": f"GPU inspection failed: {type(exc).__name__}: {exc}",
            "cuda_available": False,
        }


def _select_rows(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return frame.sample(n=maximum, random_state=seed).reset_index(drop=True)


def build_training_features(
    frame: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    stride: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one answer-containing Longformer feature per source example."""
    features: list[dict[str, Any]] = []
    dropped = 0
    for row in frame.to_dict(orient="records"):
        question = str(row["question"]).strip()
        context = str(row["document"])
        answer_start = int(row["answer_start"])
        answer_end = int(row["answer_end"])
        encoded = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        selected: dict[str, Any] | None = None
        selected_distance = float("inf")
        for feature_index, offsets in enumerate(encoded["offset_mapping"]):
            sequence_ids = encoded.sequence_ids(feature_index)
            context_positions = [i for i, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
            if not context_positions:
                continue
            context_start = context_positions[0]
            context_end = context_positions[-1]
            if offsets[context_start][0] > answer_start or offsets[context_end][1] < answer_end:
                continue

            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start:
                token_start += 1
            token_start -= 1
            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= answer_end:
                token_end -= 1
            token_end += 1
            center = (context_start + context_end) / 2
            answer_center = (token_start + token_end) / 2
            distance = abs(center - answer_center)

            item = {
                "input_ids": encoded["input_ids"][feature_index],
                "attention_mask": encoded["attention_mask"][feature_index],
                "start_positions": int(token_start),
                "end_positions": int(token_end),
            }
            if "token_type_ids" in encoded:
                item["token_type_ids"] = encoded["token_type_ids"][feature_index]
            global_attention = [
                1 if sequence_id == 0 and item["attention_mask"][index] == 1 else 0
                for index, sequence_id in enumerate(sequence_ids)
            ]
            item["global_attention_mask"] = global_attention
            if distance < selected_distance:
                selected = item
                selected_distance = distance

        if selected is None:
            dropped += 1
        else:
            features.append(selected)

    summary = {
        "source_examples": int(len(frame)),
        "training_features": int(len(features)),
        "dropped_without_answer_window": int(dropped),
        "max_length": max_length,
        "stride": stride,
    }
    return features, summary


def fine_tune_longformer(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    project_root: str | Path,
    profile_name: str = "portfolio",
    base_model_id: str = "valhalla/longformer-base-4096-finetuned-squadv1",
    seed: int = 42,
) -> dict[str, Any]:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}. Choose from {sorted(PROFILES)}")

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        DefaultDataCollator,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    profile = PROFILES[profile_name]
    project_root = Path(project_root)
    output_dir = project_root / "models" / "qasper-longformer"
    logs_dir = project_root / "outputs" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    selected_train = _select_rows(train_frame, profile.train_examples, seed)
    selected_validation = _select_rows(
        validation_frame, profile.validation_examples, seed + 1
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    train_features, train_feature_summary = build_training_features(
        selected_train, tokenizer, profile.max_length, profile.stride
    )
    validation_features, validation_feature_summary = build_training_features(
        selected_validation, tokenizer, profile.max_length, profile.stride
    )
    if not train_features:
        raise ValueError("No answer-containing training features were generated.")

    model = AutoModelForQuestionAnswering.from_pretrained(base_model_id)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_cuda and not use_bf16)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = Dataset.from_list(train_features)
    validation_dataset = Dataset.from_list(validation_features)
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps" if len(validation_dataset) else "no",
        save_strategy="steps",
        learning_rate=profile.learning_rate,
        per_device_train_batch_size=profile.train_batch_size,
        per_device_eval_batch_size=profile.eval_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        num_train_epochs=profile.epochs,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=profile.logging_steps,
        save_steps=profile.save_steps,
        eval_steps=profile.save_steps,
        save_total_limit=2,
        load_best_model_at_end=bool(len(validation_dataset)),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=bool(use_cuda),
        gradient_checkpointing=True,
        auto_find_batch_size=True,
        dataloader_num_workers=0,
        report_to="none",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset if len(validation_dataset) else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    started = time.perf_counter()
    result = trainer.train()
    duration = time.perf_counter() - started
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()

    history = trainer.state.log_history
    (logs_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, default=str), encoding="utf-8"
    )
    summary = {
        "status": "completed",
        "base_model_id": base_model_id,
        "saved_model_path": str(output_dir),
        "profile": profile.to_dict(),
        "seed": seed,
        "cuda_available": bool(use_cuda),
        "gpu_name": torch.cuda.get_device_name(0) if use_cuda else None,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "training_duration_seconds": duration,
        "training_loss": float(result.training_loss),
        "global_steps": int(trainer.state.global_step),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_features": train_feature_summary,
        "validation_features": validation_feature_summary,
        "fine_tuned_by_this_project": True,
    }
    (project_root / "outputs" / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (project_root / "models" / "fine_tuned_longformer_metadata.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
