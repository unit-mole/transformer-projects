"""LoRA/PEFT training workflow for FLAN-T5-small."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .instruction_dataset_builder import build_dataset_dict
from .peft_lora_config import LoraSettings, create_lora_config
from .tokenizer_utils import tokenize_batch


def train_lora(config_path: str | Path) -> dict:
    try:
        from peft import get_peft_model
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies from requirements.txt.") from exc

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["base_model_id"])
    model = get_peft_model(model, create_lora_config(LoraSettings(
        r=int(lora_cfg["r"]),
        alpha=int(lora_cfg["alpha"]),
        dropout=float(lora_cfg["dropout"]),
        target_modules=tuple(lora_cfg["target_modules"]),
    )))
    model.print_trainable_parameters()

    dataset = build_dataset_dict(data_cfg["dataset_path"])
    tokenized = dataset.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
            max_source_length=int(data_cfg["max_source_length"]),
            max_target_length=int(data_cfg["max_target_length"]),
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["epochs"]),
        weight_decay=float(train_cfg["weight_decay"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        report_to="none",
        fp16=bool(train_cfg.get("fp16", False)),
        seed=int(train_cfg["seed"]),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "base_model_id": model_cfg["base_model_id"],
        "adapter_path": str(output_dir),
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "training_metrics": train_result.metrics,
        "fine_tuning_method": "LoRA / PEFT",
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
