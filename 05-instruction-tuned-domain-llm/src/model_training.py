"""Reproducible LoRA fine-tuning workflow for FLAN-T5.

This module is designed for a single local RTX GPU. It auto-detects mixed
precision, records hardware/package metadata, saves the best LoRA adapter, and
exports trainer logs as JSON/CSV/PNG artifacts.
"""
from __future__ import annotations

import csv
import inspect
import json
import math
import os
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .config import LoraTrainingConfig, ModelConfig
from .data_preprocessing import load_jsonl, split_records, validate_and_clean_records
from .hardware_utils import HardwareProfile, detect_hardware, save_hardware_report
from .peft_lora_config import build_lora_config
from .tokenizer_utils import tokenize_batch


def _set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        pass


def resolve_training_config(
    base: LoraTrainingConfig | None = None,
    hardware: HardwareProfile | None = None,
) -> LoraTrainingConfig:
    """Merge user defaults with conservative hardware-aware batch settings."""
    cfg = base or LoraTrainingConfig()
    hw = hardware or detect_hardware()
    if not hw.cuda_available:
        return cfg
    return replace(
        cfg,
        per_device_train_batch_size=hw.train_batch_size,
        per_device_eval_batch_size=hw.eval_batch_size,
        gradient_accumulation_steps=hw.gradient_accumulation_steps,
        gradient_checkpointing=hw.gradient_checkpointing,
        dataloader_num_workers=hw.dataloader_num_workers,
    )


def _training_arguments(
    output_dir: Path,
    cfg: LoraTrainingConfig,
    has_validation: bool,
    hardware: HardwareProfile,
) -> Any:
    from transformers import Seq2SeqTrainingArguments

    use_bf16 = hardware.cuda_available and hardware.recommended_precision == "bf16"
    use_fp16 = hardware.cuda_available and hardware.recommended_precision == "fp16"
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "logging_strategy": "steps",
        "logging_steps": cfg.logging_steps,
        "logging_first_step": True,
        "save_strategy": "epoch",
        "save_total_limit": cfg.save_total_limit,
        "predict_with_generate": True,
        "generation_max_length": 256,
        "generation_num_beams": 4,
        "report_to": "none",
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "load_best_model_at_end": bool(has_validation),
        "metric_for_best_model": "eval_loss" if has_validation else None,
        "greater_is_better": False if has_validation else None,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "tf32": bool(hardware.cuda_available),
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "dataloader_pin_memory": bool(hardware.cuda_available),
        "group_by_length": cfg.group_by_length,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "label_smoothing_factor": cfg.label_smoothing_factor,
        "max_grad_norm": cfg.max_grad_norm,
        "optim": cfg.optim,
        "save_safetensors": True,
        "include_inputs_for_metrics": False,
        "remove_unused_columns": True,
    }
    parameter_names = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"
    kwargs[strategy_name] = "epoch" if has_validation else "no"
    # Keep compatibility across Transformers 4.x and 5.x.
    kwargs = {k: v for k, v in kwargs.items() if k in parameter_names and v is not None}
    return Seq2SeqTrainingArguments(**kwargs)


def _save_log_history(log_history: list[dict[str, Any]], output: Path) -> None:
    (output / "training_log_history.json").write_text(
        json.dumps(log_history, indent=2, default=str), encoding="utf-8"
    )
    keys = sorted({key for row in log_history for key in row})
    with (output / "training_log_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_history)


def _plot_training_curves(log_history: list[dict[str, Any]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    train_points = [(row.get("step"), row.get("loss")) for row in log_history if row.get("loss") is not None]
    eval_points = [(row.get("step"), row.get("eval_loss")) for row in log_history if row.get("eval_loss") is not None]
    if not train_points and not eval_points:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if train_points:
        ax.plot([p[0] for p in train_points], [p[1] for p in train_points], marker="o", label="Training loss")
    if eval_points:
        ax.plot([p[0] for p in eval_points], [p[1] for p in eval_points], marker="s", label="Validation loss")
    ax.set_title("FLAN-T5 LoRA Training Curve")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curve.png", dpi=160)
    plt.close(fig)


def train_lora_adapter(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[LoraTrainingConfig] = None,
    hardware_profile: Optional[HardwareProfile] = None,
    resume_from_checkpoint: str | bool | None = None,
) -> Dict[str, object]:
    """Train, validate, and save the best LoRA adapter plus reproducibility artifacts."""
    try:
        import torch
        import transformers
        from datasets import Dataset
        from peft import get_peft_model
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
        )
    except ImportError as exc:
        raise ImportError(
            "Install requirements-training.txt before running LoRA training."
        ) from exc

    hw = hardware_profile or detect_hardware()
    if not hw.cuda_available:
        raise RuntimeError(
            "CUDA was not detected. This quality training workflow requires an NVIDIA GPU. "
            "Verify your CUDA-enabled PyTorch installation with torch.cuda.is_available()."
        )

    mcfg = model_config or ModelConfig(base_model_id=hw.recommended_model_id)
    tcfg = resolve_training_config(training_config, hw)
    _set_reproducibility(tcfg.seed)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_hardware_report(output / "hardware_report.json", hw)

    raw_records = load_jsonl(dataset_path)
    cleaned, report = validate_and_clean_records(raw_records)
    splits = split_records(cleaned)
    if not splits["train"]:
        raise ValueError("No valid training records were found.")
    if not splits["validation"]:
        raise ValueError("A validation split is required for early stopping and best-checkpoint selection.")

    tokenizer = AutoTokenizer.from_pretrained(mcfg.base_model_id, use_fast=True)
    model_dtype = torch.bfloat16 if hw.recommended_precision == "bf16" else torch.float16
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        mcfg.base_model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    if tcfg.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    model = get_peft_model(base_model, build_lora_config(tcfg))
    if tcfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_dataset = Dataset.from_list(splits["train"])
    eval_dataset = Dataset.from_list(splits["validation"])
    test_dataset = Dataset.from_list(splits["test"]) if splits["test"] else None

    def preprocess(batch: Dict[str, list[Any]]) -> Dict[str, Any]:
        return tokenize_batch(
            batch,
            tokenizer,
            max_input_length=mcfg.max_input_length,
            max_target_length=mcfg.max_target_length,
        )

    tokenized_train = train_dataset.map(
        preprocess, batched=True, remove_columns=train_dataset.column_names, desc="Tokenizing train split"
    )
    tokenized_eval = eval_dataset.map(
        preprocess, batched=True, remove_columns=eval_dataset.column_names, desc="Tokenizing validation split"
    )
    tokenized_test = (
        test_dataset.map(preprocess, batched=True, remove_columns=test_dataset.column_names, desc="Tokenizing test split")
        if test_dataset is not None
        else None
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    args = _training_arguments(output, tcfg, True, hw)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_eval,
        "data_collator": data_collator,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=tcfg.early_stopping_patience)],
    }
    trainer_parameters = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    model.print_trainable_parameters()

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    test_metrics = trainer.evaluate(tokenized_test, metric_key_prefix="test") if tokenized_test is not None else {}

    adapter_dir = output / "lora_adapter"
    tokenizer_dir = output / "tokenizer"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(tokenizer_dir)
    trainer.save_state()

    log_history = list(trainer.state.log_history)
    _save_log_history(log_history, output)
    _plot_training_curves(log_history, output)

    best_eval_loss = validation_metrics.get("validation_loss")
    metadata: Dict[str, object] = {
        "status": "completed",
        "base_model": mcfg.base_model_id,
        "fine_tuning_method": "LoRA/PEFT",
        "adapter_path": str(adapter_dir),
        "tokenizer_path": str(tokenizer_dir),
        "dataset_path": str(dataset_path),
        "dataset_validation": report.to_dict(),
        "dataset_split_sizes": {name: len(records) for name, records in splits.items()},
        "model_config": asdict(mcfg),
        "training_config": asdict(tcfg),
        "hardware": hw.to_dict(),
        "package_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "train_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in train_result.metrics.items()},
        "validation_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in validation_metrics.items()},
        "test_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in test_metrics.items()},
        "validation_perplexity": round(math.exp(float(best_eval_loss)), 6)
        if isinstance(best_eval_loss, (int, float)) and best_eval_loss < 20
        else None,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_percentage": round(100 * trainable_parameters / total_parameters, 6),
    }
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (adapter_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata
