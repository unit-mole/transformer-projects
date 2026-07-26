from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset_loader import load_iitb_dataframe


BASE_MODELS = {
    "en_hi": "Helsinki-NLP/opus-mt-en-hi",
    "hi_en": "Helsinki-NLP/opus-mt-hi-en",
}


def fine_tune_marian(
    *,
    direction: str,
    dataset_name: str = "cfilt/iitb-english-hindi",
    train_split: str = "train[:10000]",
    validation_split: str = "validation",
    output_dir: str | Path,
    epochs: float = 1.0,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_source_length: int = 128,
    max_target_length: int = 128,
) -> dict[str, Any]:
    """Optional fine-tuning entry point. Not called by the demo."""
    if direction not in BASE_MODELS:
        raise ValueError("direction must be 'en_hi' or 'hi_en'.")

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before fine-tuning.") from exc

    train_df = load_iitb_dataframe(train_split, dataset_name=dataset_name)
    validation_df = load_iitb_dataframe(validation_split, dataset_name=dataset_name)

    source_column, target_column = (
        ("english", "hindi") if direction == "en_hi" else ("hindi", "english")
    )
    model_id = BASE_MODELS[direction]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        inputs = tokenizer(
            batch[source_column],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[target_column],
            max_length=max_target_length,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_dataset = Dataset.from_pandas(
        train_df[[source_column, target_column]],
        preserve_index=False,
    ).map(tokenize_batch, batched=True, remove_columns=[source_column, target_column])
    validation_dataset = Dataset.from_pandas(
        validation_df[[source_column, target_column]],
        preserve_index=False,
    ).map(tokenize_batch, batched=True, remove_columns=[source_column, target_column])

    destination = Path(output_dir)
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(destination),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        predict_with_generate=True,
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
        fp16=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))

    return {
        "direction": direction,
        "base_model": model_id,
        "output_dir": str(destination),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
    }
