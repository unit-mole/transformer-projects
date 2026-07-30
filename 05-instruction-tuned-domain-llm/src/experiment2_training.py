"""Experiment 2 LoRA training with safe FLAN-T5 checkpoint loading."""
from __future__ import annotations

import csv
import inspect
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ModelConfig
from .data_preprocessing import load_jsonl, split_records, validate_and_clean_records
from .hardware_utils import HardwareProfile, detect_hardware, save_hardware_report
from .tokenizer_utils import tokenize_batch


@dataclass(frozen=True)
class Experiment2TrainingConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q", "v")
    learning_rate: float = 5e-5
    num_train_epochs: float = 5.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 20
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_total_limit: int = 2
    seed: int = 52
    lr_scheduler_type: str = "cosine"
    label_smoothing_factor: float = 0.0
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    group_by_length: bool = True
    optim: str = "adamw_torch"


def _set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def resolve_experiment2_config(
    config: Experiment2TrainingConfig | None,
    hardware: HardwareProfile,
) -> Experiment2TrainingConfig:
    cfg = config or Experiment2TrainingConfig()
    if not hardware.cuda_available:
        return cfg
    return replace(
        cfg,
        per_device_train_batch_size=hardware.train_batch_size,
        per_device_eval_batch_size=hardware.eval_batch_size,
        gradient_accumulation_steps=hardware.gradient_accumulation_steps,
        gradient_checkpointing=hardware.gradient_checkpointing,
        dataloader_num_workers=hardware.dataloader_num_workers,
    )


def _training_arguments(
    output_dir: Path,
    cfg: Experiment2TrainingConfig,
    hardware: HardwareProfile,
) -> Any:
    from transformers import Seq2SeqTrainingArguments

    parameters = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
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
        "warmup_steps": cfg.warmup_steps,
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
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
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
        "remove_unused_columns": True,
    }
    strategy_name = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[strategy_name] = "epoch"
    return Seq2SeqTrainingArguments(
        **{key: value for key, value in kwargs.items() if key in parameters}
    )


def _verify_t5_checkpoint_loading(
    model: Any,
    loading_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify the pretrained T5 shared embedding structure.

    T5 uses one shared encoder/decoder input embedding matrix. Some Transformers
    versions report encoder.embed_tokens.weight and decoder.embed_tokens.weight
    as missing alias keys even when shared.weight loaded correctly. Those aliases
    are allowed; any other missing checkpoint parameter stops training.
    """
    import torch

    missing_keys = {str(key) for key in loading_info.get("missing_keys", [])}
    expected_aliases = {
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    }
    if getattr(model.config, "tie_word_embeddings", True):
        expected_aliases.add("lm_head.weight")

    critical_missing = sorted(
        key for key in missing_keys if key not in expected_aliases
    )
    if "shared.weight" in missing_keys:
        raise RuntimeError(
            "FLAN-T5 checkpoint loading failed: shared.weight is missing."
        )
    if critical_missing:
        raise RuntimeError(
            "Unexpected missing FLAN-T5 checkpoint keys: "
            + ", ".join(critical_missing)
        )

    required = ("shared", "encoder", "decoder", "lm_head")
    absent = [name for name in required if not hasattr(model, name)]
    if absent:
        raise RuntimeError(
            "Loaded model is missing required T5 attributes: "
            + ", ".join(absent)
        )

    shared_weight = model.shared.weight
    encoder_weight = model.encoder.embed_tokens.weight
    decoder_weight = model.decoder.embed_tokens.weight

    if shared_weight.device.type == "meta":
        raise RuntimeError("FLAN-T5 shared embeddings remain on the meta device.")
    if not torch.isfinite(shared_weight.detach().float()).all().item():
        raise RuntimeError("FLAN-T5 shared embeddings contain non-finite values.")

    encoder_shared = encoder_weight.data_ptr() == shared_weight.data_ptr()
    decoder_shared = decoder_weight.data_ptr() == shared_weight.data_ptr()
    lm_head_shared = model.lm_head.weight.data_ptr() == shared_weight.data_ptr()

    if not encoder_shared or not decoder_shared:
        raise RuntimeError(
            "T5 embedding integrity check failed: encoder and decoder must use "
            "the pretrained shared embedding matrix."
        )

    verification = {
        "shared_weight_loaded": True,
        "encoder_uses_shared_weight": encoder_shared,
        "decoder_uses_shared_weight": decoder_shared,
        "lm_head_uses_shared_weight": lm_head_shared,
        "tie_word_embeddings_config": bool(
            getattr(model.config, "tie_word_embeddings", True)
        ),
        "expected_alias_missing_keys": sorted(
            missing_keys.intersection(expected_aliases)
        ),
        "unexpected_missing_keys": critical_missing,
        "unexpected_keys": [
            str(key) for key in loading_info.get("unexpected_keys", [])
        ],
        "mismatched_keys": [
            str(key) for key in loading_info.get("mismatched_keys", [])
        ],
        "error_messages": [
            str(message) for message in loading_info.get("error_msgs", [])
        ],
    }

    print("=" * 72)
    print("FLAN-T5 PRETRAINED CHECKPOINT VERIFICATION")
    print("=" * 72)
    print("Shared embedding loaded       : True")
    print(f"Encoder uses shared embedding : {encoder_shared}")
    print(f"Decoder uses shared embedding : {decoder_shared}")
    print(f"LM head uses shared embedding : {lm_head_shared}")
    print(
        "Config tie_word_embeddings   : "
        f"{verification['tie_word_embeddings_config']}"
    )
    if verification["expected_alias_missing_keys"]:
        print(
            "Expected T5 alias keys       : "
            + ", ".join(verification["expected_alias_missing_keys"])
        )
    print("Pretrained FLAN-T5 embedding verification passed.")

    return verification


def _save_logs(log_history: list[dict[str, Any]], output: Path) -> None:
    (output / "training_log_history.json").write_text(
        json.dumps(log_history, indent=2, default=str), encoding="utf-8"
    )
    keys = sorted({key for row in log_history for key in row})
    with (output / "training_log_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_history)


def _plot_curves(log_history: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    train = [(r.get("step"), r.get("loss")) for r in log_history if r.get("loss") is not None]
    valid = [(r.get("step"), r.get("eval_loss")) for r in log_history if r.get("eval_loss") is not None]
    if not train and not valid:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    if train:
        ax.plot([p[0] for p in train], [p[1] for p in train], marker="o", label="Training loss")
    if valid:
        ax.plot([p[0] for p in valid], [p[1] for p in valid], marker="s", label="Validation loss")
    ax.set_title("Experiment 2 — FLAN-T5-base LoRA Training")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curve.png", dpi=170)
    plt.close(fig)


def train_experiment2_adapter(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[Experiment2TrainingConfig] = None,
    hardware_profile: Optional[HardwareProfile] = None,
    resume_from_checkpoint: str | bool | None = None,
) -> Dict[str, Any]:
    """Train the second adapter and save an auditable Experiment 2 artifact set."""
    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
    )

    hardware = hardware_profile or detect_hardware()
    if not hardware.cuda_available:
        raise RuntimeError("Experiment 2 requires a CUDA-enabled NVIDIA GPU.")
    mcfg = model_config or ModelConfig(base_model_id="google/flan-t5-base")
    tcfg = resolve_experiment2_config(training_config, hardware)
    _set_reproducibility(tcfg.seed)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_hardware_report(output / "hardware_report.json", hardware)

    raw = load_jsonl(dataset_path)
    cleaned, validation_report = validate_and_clean_records(raw, min_output_words=12, max_output_words=300)
    splits = split_records(cleaned)
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        raise ValueError("Version 3 must contain train, validation, and test splits.")

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg.base_model_id,
        use_fast=True,
        trust_remote_code=mcfg.trust_remote_code,
    )
    dtype = torch.bfloat16 if hardware.recommended_precision == "bf16" else torch.float16

    # Preserve the pretrained FLAN-T5 configuration. Do not force
    # tie_word_embeddings=False before loading the checkpoint.
    base_model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
        mcfg.base_model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=mcfg.trust_remote_code,
        output_loading_info=True,
    )
    loading_verification = _verify_t5_checkpoint_loading(base_model, loading_info)
    (output / "model_loading_report.json").write_text(
        json.dumps(
            {
                "loading_info": {
                    key: [str(item) for item in value]
                    if isinstance(value, (list, tuple))
                    else str(value)
                    for key, value in loading_info.items()
                },
                "verification": loading_verification,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    base_model.config.use_cache = False
    if tcfg.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=tcfg.r,
        lora_alpha=tcfg.lora_alpha,
        lora_dropout=tcfg.lora_dropout,
        target_modules=list(tcfg.target_modules),
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    if tcfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    datasets = {name: Dataset.from_list(rows) for name, rows in splits.items()}

    def preprocess(batch: Dict[str, list[Any]]) -> Dict[str, Any]:
        return tokenize_batch(
            batch,
            tokenizer,
            max_input_length=mcfg.max_input_length,
            max_target_length=mcfg.max_target_length,
        )

    tokenized = {
        name: dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing Experiment 2 {name} split",
        )
        for name, dataset in datasets.items()
    }

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    args = _training_arguments(output, tcfg, hardware)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": collator,
        "callbacks": [
            EarlyStoppingCallback(early_stopping_patience=tcfg.early_stopping_patience)
        ],
    }
    parameters = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    model.print_trainable_parameters()

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.remove_callback(EarlyStoppingCallback)
    validation_raw = trainer.evaluate(tokenized["validation"])
    test_raw = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    adapter_dir = output / "lora_adapter"
    tokenizer_dir = output / "tokenizer"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(tokenizer_dir)
    trainer.save_state()

    log_history = list(trainer.state.log_history)
    _save_logs(log_history, output)
    _plot_curves(log_history, output)

    validation_loss = validation_raw.get("eval_loss")
    validation_metrics = {
        "validation_loss": float(validation_loss) if validation_loss is not None else None,
        **{k: float(v) if isinstance(v, (int, float)) else v for k, v in validation_raw.items()},
    }
    test_metrics = {
        "test_loss": float(test_raw.get("test_loss")) if test_raw.get("test_loss") is not None else None,
        **{k: float(v) if isinstance(v, (int, float)) else v for k, v in test_raw.items()},
    }
    metadata: Dict[str, Any] = {
        "status": "completed",
        "experiment": "experiment_2_dataset_quality_upgrade",
        "base_model": mcfg.base_model_id,
        "fine_tuning_method": "LoRA/PEFT",
        "adapter_path": str(adapter_dir.resolve()),
        "tokenizer_path": str(tokenizer_dir.resolve()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_validation": validation_report.to_dict(),
        "dataset_split_sizes": {name: len(rows) for name, rows in splits.items()},
        "model_config": asdict(mcfg),
        "training_config": asdict(tcfg),
        "hardware": hardware.to_dict(),
        "package_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "model_loading_verification": loading_verification,
        "train_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in train_result.metrics.items()
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_perplexity": (
            round(math.exp(float(validation_loss)), 6)
            if isinstance(validation_loss, (int, float)) and validation_loss < 20
            else None
        ),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_percentage": round(100 * trainable_parameters / total_parameters, 6),
        "experiment_2_training_fixes": [
            "warmup_steps replaces deprecated warmup_ratio",
            "the original FLAN-T5 tie_word_embeddings configuration is preserved",
            "shared encoder/decoder embedding aliases are verified after loading",
            "unexpected missing checkpoint parameters stop training",
            "early stopping is removed before final validation/test evaluation",
        ],
    }
    (output / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    (adapter_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return metadata
