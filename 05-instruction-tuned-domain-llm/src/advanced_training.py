"""GPU-aware LoRA training for the portfolio-scale Project 05 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .experiment_utils import choose_precision, count_parameters, hardware_info, save_json, set_reproducibility, utc_run_id
from .instruction_dataset_builder import build_dataset_dict
from .peft_lora_config import LoraSettings, create_lora_config
from .tokenizer_utils import tokenize_batch


def train_portfolio_lora(
    config_path: str | Path,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from peft import get_peft_model
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install requirements-training.txt before training.") from exc

    config_path = Path(config_path).resolve()
    project_dir = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    experiment_cfg = config.get("experiment", {})

    seed = int(train_cfg.get("seed", 42))
    set_reproducibility(seed, deterministic=bool(experiment_cfg.get("full_determinism", False)))
    precision = choose_precision(str(train_cfg.get("precision", "auto")))
    if precision["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_id = str(experiment_cfg.get("run_id") or utc_run_id())
    output_root = Path(train_cfg["output_dir"])
    if not output_root.is_absolute():
        output_root = project_dir / output_root
    dataset_path = Path(data_cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = project_dir / dataset_path
    run_dir = output_root / run_id
    adapter_dir = run_dir / "adapter"
    logs_dir = run_dir / "logs"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["base_model_id"])
    lora_settings = LoraSettings(
        r=int(lora_cfg.get("r", 16)),
        alpha=int(lora_cfg.get("alpha", 32)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=tuple(lora_cfg.get("target_modules", ["q", "v"])),
    )
    model = get_peft_model(model, create_lora_config(lora_settings))
    model.config.use_cache = False
    parameter_info = count_parameters(model)
    model.print_trainable_parameters()

    dataset = build_dataset_dict(dataset_path)
    tokenized = dataset.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
            max_source_length=int(data_cfg.get("max_source_length", 384)),
            max_target_length=int(data_cfg.get("max_target_length", 192)),
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing instruction dataset",
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=run_id,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        per_device_train_batch_size=int(train_cfg.get("train_batch_size", 4)),
        per_device_eval_batch_size=int(train_cfg.get("eval_batch_size", 8)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
        num_train_epochs=float(train_cfg.get("epochs", 8)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        logging_strategy="steps",
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        logging_first_step=True,
        logging_dir=str(logs_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        report_to="none",
        bf16=bool(precision["bf16"]),
        fp16=bool(precision["fp16"]),
        tf32=bool(precision["tf32"]),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        optim=str(train_cfg.get("optim", "adamw_torch")),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        seed=seed,
        data_seed=seed,
        full_determinism=bool(experiment_cfg.get("full_determinism", False)),
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(train_cfg.get("early_stopping_patience", 2)))],
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_state()
    model.config.use_cache = True

    log_history = trainer.state.log_history
    save_json(log_history, run_dir / "training_log_history.json")
    save_json(train_result.metrics, run_dir / "training_metrics.json")
    save_json(eval_metrics, run_dir / "validation_metrics.json")

    metadata = {
        "status": "completed",
        "run_id": run_id,
        "base_model_id": model_cfg["base_model_id"],
        "adapter_path": str(adapter_dir),
        "dataset_path": str(dataset_path),
        "dataset_sizes": {split: len(dataset[split]) for split in dataset},
        "lora": {
            "r": lora_settings.r,
            "alpha": lora_settings.alpha,
            "dropout": lora_settings.dropout,
            "target_modules": list(lora_settings.target_modules),
        },
        "parameters": parameter_info,
        "precision": precision,
        "training_metrics": train_result.metrics,
        "validation_metrics": eval_metrics,
        "hardware": hardware_info(),
        "config_path": str(config_path),
    }
    save_json(metadata, run_dir / "training_metadata.json")
    save_json(metadata, output_root / "latest_training_metadata.json")
    (output_root / "LATEST_RUN.txt").write_text(run_id, encoding="utf-8")
    return metadata


def plot_training_history(log_history: list[dict[str, Any]], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    train_steps = [item["step"] for item in log_history if "loss" in item]
    train_loss = [item["loss"] for item in log_history if "loss" in item]
    eval_steps = [item["step"] for item in log_history if "eval_loss" in item]
    eval_loss = [item["eval_loss"] for item in log_history if "eval_loss" in item]
    fig, ax = plt.subplots(figsize=(9, 5))
    if train_steps:
        ax.plot(train_steps, train_loss, marker="o", label="Training loss")
    if eval_steps:
        ax.plot(eval_steps, eval_loss, marker="o", label="Validation loss")
    ax.set_title("LoRA Training and Validation Loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
