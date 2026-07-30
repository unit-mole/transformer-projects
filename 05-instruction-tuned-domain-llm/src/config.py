"""Central configuration for the ML/Data Science instruction-tuned assistant."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"


@dataclass(frozen=True)
class ModelConfig:
    # FLAN-T5-base is the quality preset. Override BASE_MODEL_ID with
    # google/flan-t5-small when a lower-memory or faster CPU demo is required.
    base_model_id: str = os.getenv("BASE_MODEL_ID", "google/flan-t5-base")
    adapter_id: str = os.getenv("ADAPTER_ID", "")
    local_adapter_path: str = os.getenv(
        "LOCAL_ADAPTER_PATH", str(MODELS_DIR / "lora_adapter")
    )
    max_input_length: int = int(os.getenv("MAX_INPUT_LENGTH", "512"))
    max_target_length: int = int(os.getenv("MAX_TARGET_LENGTH", "256"))
    device: str = os.getenv("MODEL_DEVICE", "auto")
    torch_dtype: str = os.getenv("TORCH_DTYPE", "auto")
    trust_remote_code: bool = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 220
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.12
    do_sample: bool = False
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3


@dataclass(frozen=True)
class LoraTrainingConfig:
    # A moderately expressive adapter for FLAN-T5-base. q/v are the attention
    # projections documented by PEFT for T5-style sequence-to-sequence models.
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q", "v")
    learning_rate: float = 1e-4
    num_train_epochs: float = 6.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.08
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_total_limit: int = 2
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    label_smoothing_factor: float = 0.05
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    group_by_length: bool = True
    optim: str = "adamw_torch"


@dataclass(frozen=True)
class DatasetGenerationConfig:
    teacher_model_id: str = os.getenv(
        "TEACHER_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"
    )
    target_examples: int = int(os.getenv("TARGET_DATASET_EXAMPLES", "600"))
    temperature: float = 0.55
    top_p: float = 0.9
    max_new_tokens: int = 1800
    examples_per_topic: int = 8
    seed: int = 42
    duplicate_similarity_threshold: float = 0.88
    benchmark_leakage_threshold: float = 0.78


@dataclass(frozen=True)
class EvaluationConfig:
    benchmark_path: str = str(DATA_DIR / "benchmark_prompts_v2.jsonl")
    bertscore_model_type: str = "roberta-large"
    semantic_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    seed: int = 42
    include_bertscore: bool = True
    include_rouge: bool = True
    include_semantic_similarity: bool = True


def model_metadata() -> Dict[str, Any]:
    return {
        "model": asdict(ModelConfig()),
        "generation": asdict(GenerationConfig()),
        "lora_training": asdict(LoraTrainingConfig()),
        "dataset_generation": asdict(DatasetGenerationConfig()),
        "evaluation": asdict(EvaluationConfig()),
        "task": "ML and Data Science educational instruction following",
        "fine_tuning_method": "LoRA through Hugging Face PEFT",
        "status": "adapter_not_included_train_or_set_ADAPTER_ID",
    }
