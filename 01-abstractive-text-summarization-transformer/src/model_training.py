from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "sshleifer/distilbart-cnn-12-6"
    output_dir: str = "models/transformer_summarization_model"
    tokenizer_dir: str = "models/tokenizer"
    max_input_tokens: int = 768
    max_target_tokens: int = 128
    learning_rate: float = 2e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    seed: int = 42


def train_model(train_frame: Any, validation_frame: Any, config: TrainingConfig) -> dict[str, Any]:
    """Fine-tune DistilBART. A CUDA environment is strongly recommended."""
    try:
        import evaluate
        import numpy as np
        from datasets import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before training.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    train_dataset = Dataset.from_pandas(train_frame[["article", "reference_summary"]])
    validation_dataset = Dataset.from_pandas(
        validation_frame[["article", "reference_summary"]]
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch["article"],
            max_length=config.max_input_tokens,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["reference_summary"],
            max_length=config.max_target_tokens,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(tokenize_batch, batched=True, remove_columns=train_dataset.column_names)
    validation_tokenized = validation_dataset.map(
        tokenize_batch, batched=True, remove_columns=validation_dataset.column_names
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(evaluation_prediction: Any) -> dict[str, float]:
        predictions, labels = evaluation_prediction
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return rouge.compute(predictions=decoded_predictions, references=decoded_labels)

    arguments = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=config.max_target_tokens,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        report_to="none",
        seed=config.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_tokenized,
        eval_dataset=validation_tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    evaluation = trainer.evaluate()
    trainer.save_model(config.output_dir)
    Path(config.tokenizer_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(config.tokenizer_dir)
    return {"train_metrics": train_result.metrics, "evaluation_metrics": evaluation}
