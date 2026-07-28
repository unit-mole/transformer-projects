from __future__ import annotations

import json
import math
import os
import platform
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

DIRECTIONS = {
    "en_hi": ("english", "hindi"),
    "hi_en": ("hindi", "english"),
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
REPEATED_TOKEN_RE = re.compile(r"\b(\S+)(?:\s+\1){2,}\b", re.IGNORECASE)

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


@dataclass(frozen=True)
class HardwareProfile:
    device: str
    gpu_name: str | None
    gpu_memory_gb: float | None
    bf16_supported: bool
    recommended_train_batch_size: int
    recommended_eval_batch_size: int
    gradient_accumulation_steps: int


def project_root_from(path: str | Path | None = None) -> Path:
    candidate = Path(path or Path.cwd()).resolve()
    if candidate.name == "notebooks":
        candidate = candidate.parent
    if (candidate / "configs" / "portfolio_evaluation.yaml").exists():
        return candidate
    for parent in candidate.parents:
        if (parent / "configs" / "portfolio_evaluation.yaml").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate the Project 02 root containing configs/portfolio_evaluation.yaml."
    )


def load_config(
    project_root: str | Path,
    config_path: str | Path = "configs/portfolio_evaluation.yaml",
    profile: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    path = root / config_path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    active = profile or config.get("active_profile", "portfolio")
    if active not in config["profiles"]:
        raise ValueError(f"Unknown profile '{active}'. Available: {sorted(config['profiles'])}")
    config["active_profile"] = active
    config["profile"] = dict(config["profiles"][active])
    config["project_root"] = str(root.resolve())
    return config


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def detect_hardware() -> HardwareProfile:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install a CUDA-enabled PyTorch build before running the notebook."
        ) from exc

    if not torch.cuda.is_available():
        return HardwareProfile(
            device="cpu",
            gpu_name=None,
            gpu_memory_gb=None,
            bf16_supported=False,
            recommended_train_batch_size=2,
            recommended_eval_batch_size=4,
            gradient_accumulation_steps=8,
        )

    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / (1024**3)
    if memory_gb >= 20:
        train_batch, eval_batch, grad_accum = 12, 16, 2
    elif memory_gb >= 14:
        train_batch, eval_batch, grad_accum = 8, 12, 2
    elif memory_gb >= 10:
        train_batch, eval_batch, grad_accum = 4, 8, 4
    elif memory_gb >= 7:
        train_batch, eval_batch, grad_accum = 2, 4, 8
    else:
        train_batch, eval_batch, grad_accum = 1, 2, 16

    bf16_supported = bool(
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    return HardwareProfile(
        device="cuda",
        gpu_name=props.name,
        gpu_memory_gb=round(memory_gb, 2),
        bf16_supported=bf16_supported,
        recommended_train_batch_size=train_batch,
        recommended_eval_batch_size=eval_batch,
        gradient_accumulation_steps=grad_accum,
    )


def collect_environment(hardware: HardwareProfile) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": hardware.device,
        "gpu_name": hardware.gpu_name,
        "gpu_memory_gb": hardware.gpu_memory_gb,
        "bf16_supported": hardware.bf16_supported,
    }
    for module_name in ["torch", "transformers", "datasets", "sacrebleu", "pandas", "numpy"]:
        try:
            module = __import__(module_name)
            result[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            result[module_name] = None
    return result


def _clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _extract_pair(row: dict[str, Any], dataset_cfg: dict[str, Any]) -> tuple[str, str]:
    translation = row.get(dataset_cfg["translation_column"], {})
    if not isinstance(translation, dict):
        return "", ""
    return (
        _clean_text(translation.get(dataset_cfg["english_key"], "")),
        _clean_text(translation.get(dataset_cfg["hindi_key"], "")),
    )


def _valid_pair(english: str, hindi: str, dataset_cfg: dict[str, Any]) -> bool:
    minimum = int(dataset_cfg["min_characters"])
    maximum = int(dataset_cfg["max_characters"])
    return (
        minimum <= len(english) <= maximum
        and minimum <= len(hindi) <= maximum
        and bool(LATIN_RE.search(english))
        and bool(DEVANAGARI_RE.search(hindi))
    )


def _dataset_to_dataframe(dataset: Iterable[dict[str, Any]], dataset_cfg: dict[str, Any]) -> pd.DataFrame:
    pairs: list[dict[str, str]] = []
    for row in dataset:
        english, hindi = _extract_pair(row, dataset_cfg)
        if _valid_pair(english, hindi, dataset_cfg):
            pairs.append({"english": english, "hindi": hindi})
    frame = pd.DataFrame(pairs)
    if dataset_cfg.get("remove_duplicates", True) and not frame.empty:
        frame = frame.drop_duplicates(subset=["english", "hindi"])
    return frame.reset_index(drop=True)


def load_and_prepare_dataset(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements-evaluation.txt before loading the dataset.") from exc

    dataset_cfg = config["dataset"]
    profile = config["profile"]
    seed = int(config["seed"])
    dataset_id = dataset_cfg["id"]
    revision = dataset_cfg.get("revision", "main")

    requested = {
        "train": (dataset_cfg["train_split"], int(profile["train_pairs"])),
        "validation": (dataset_cfg["validation_split"], int(profile["validation_pairs"])),
        "test": (dataset_cfg["test_split"], int(profile["test_pairs"])),
    }

    prepared: dict[str, pd.DataFrame] = {}
    for name, (split_name, limit) in requested.items():
        split = load_dataset(dataset_id, split=split_name, revision=revision)
        raw_total = len(split)
        candidate_limit = min(raw_total, max(limit * 2, limit + 100))
        if raw_total > candidate_limit:
            split = split.shuffle(seed=seed).select(range(candidate_limit))
        frame = _dataset_to_dataframe(split, dataset_cfg).head(limit)
        if len(frame) < limit and candidate_limit < raw_total:
            raise RuntimeError(
                f"Only {len(frame)} valid pairs remained for {name}; {limit} were requested. "
                "Reduce the profile size or relax the cleaning limits."
            )
        if frame.empty:
            raise RuntimeError(f"No valid pairs remained for the {name} split.")
        prepared[name] = frame.reset_index(drop=True)

    return prepared


def save_prepared_dataset(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, str]:
    root = Path(config["project_root"])
    output_root = root / config["outputs"]["root"]
    dataset_dir = output_root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for split_name, frame in frames.items():
        path = dataset_dir / f"{split_name}_pairs.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[split_name] = str(path.relative_to(root))

    try:
        from huggingface_hub import HfApi

        resolved_dataset_sha = HfApi().dataset_info(
            config["dataset"]["id"],
            revision=config["dataset"].get("revision", "main"),
        ).sha
    except Exception:
        resolved_dataset_sha = None

    manifest = {
        "status": "prepared",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": config["dataset"]["id"],
        "dataset_revision": config["dataset"].get("revision", "main"),
        "resolved_dataset_sha": resolved_dataset_sha,
        "license_note": "IIT Bombay corpus: non-commercial educational/research use; verify source licenses.",
        "profile": config["active_profile"],
        "seed": config["seed"],
        "splits": {name: int(len(frame)) for name, frame in frames.items()},
        "columns": ["english", "hindi"],
        "environment": environment,
        "files": paths,
    }
    manifest_path = output_root / "dataset_manifest.json"
    _write_json(manifest_path, manifest)
    paths["manifest"] = str(manifest_path.relative_to(root))
    return paths


def load_prepared_dataset(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    root = Path(config["project_root"])
    dataset_dir = root / config["outputs"]["root"] / "datasets"
    required = {
        name: dataset_dir / f"{name}_pairs.csv"
        for name in ["train", "validation", "test"]
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Prepared dataset files are missing: {missing}")
    return {name: pd.read_csv(path) for name, path in required.items()}


def _load_model(model_ref: str, device: str, half_precision: bool) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the evaluation dependencies before loading models.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ref)
    model.to(device)
    if device == "cuda" and half_precision:
        model.half()
    model.eval()
    return torch, tokenizer, model


def _synchronize(torch_module: Any, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()


def _normalized_numbers(text: str) -> list[str]:
    normalized = text.translate(DEVANAGARI_DIGITS)
    return [match.replace(",", "") for match in NUMBER_RE.findall(normalized)]


def _script_ratios(text: str) -> dict[str, float]:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return {"latin": 0.0, "devanagari": 0.0}
    latin = sum(bool(LATIN_RE.fullmatch(char)) for char in letters)
    devanagari = sum(bool(DEVANAGARI_RE.fullmatch(char)) for char in letters)
    total = len(letters)
    return {"latin": latin / total, "devanagari": devanagari / total}


def _confidence_from_sequence_score(score: float | None) -> float | None:
    if score is None or not math.isfinite(score):
        return None
    return round(float(np.clip(math.exp(score), 0.0, 1.0)), 6)


def generate_predictions(
    dataframe: pd.DataFrame,
    *,
    model_ref: str,
    direction: str,
    hardware: HardwareProfile,
    config: dict[str, Any],
    system_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unsupported direction: {direction}")
    source_column, target_column = DIRECTIONS[direction]
    inference_cfg = config["inference"]
    batch_size = (
        hardware.recommended_eval_batch_size
        if inference_cfg.get("batch_size") == "auto"
        else int(inference_cfg["batch_size"])
    )
    half_precision = bool(inference_cfg.get("use_fp16_on_cuda", True))
    torch, tokenizer, model = _load_model(model_ref, hardware.device, half_precision)

    if hardware.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sources = dataframe[source_column].astype(str).tolist()
    references = dataframe[target_column].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    warmup_batches = int(inference_cfg.get("warmup_batches", 1))

    for batch_index, start_index in enumerate(range(0, len(sources), batch_size)):
        batch_sources = sources[start_index : start_index + batch_size]
        batch_refs = references[start_index : start_index + batch_size]
        encoded = tokenizer(
            batch_sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(config["preprocessing"]["max_source_length"]),
        )
        encoded = {key: value.to(hardware.device) for key, value in encoded.items()}
        generation_kwargs = {
            "num_beams": int(inference_cfg["num_beams"]),
            "max_new_tokens": int(inference_cfg["max_new_tokens"]),
            "early_stopping": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        if batch_index < warmup_batches:
            with torch.inference_mode():
                _ = model.generate(**encoded, **generation_kwargs)
            _synchronize(torch, hardware.device)

        _synchronize(torch, hardware.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        _synchronize(torch, hardware.device)
        elapsed = time.perf_counter() - started

        decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
        sequence_scores = getattr(generated, "sequences_scores", None)
        for offset, (source, reference, prediction) in enumerate(
            zip(batch_sources, batch_refs, decoded)
        ):
            sequence_score = None
            if sequence_scores is not None and len(sequence_scores) > offset:
                sequence_score = float(sequence_scores[offset].detach().cpu().item())
            source_ids = encoded["input_ids"][offset]
            output_ids = generated.sequences[offset]
            source_tokens = int((source_ids != tokenizer.pad_token_id).sum().item())
            output_tokens = int((output_ids != tokenizer.pad_token_id).sum().item())
            ratios = _script_ratios(prediction)
            rows.append(
                {
                    "system": system_name,
                    "direction": direction,
                    "model_ref": model_ref,
                    "source_text": source,
                    "reference_translation": reference,
                    "predicted_translation": prediction.strip(),
                    "source_characters": len(source),
                    "reference_characters": len(reference),
                    "prediction_characters": len(prediction.strip()),
                    "source_tokens": source_tokens,
                    "output_tokens": output_tokens,
                    "latency_seconds": elapsed / max(len(batch_sources), 1),
                    "sequence_score": sequence_score,
                    "confidence_proxy": _confidence_from_sequence_score(sequence_score),
                    "numbers_match": _normalized_numbers(source) == _normalized_numbers(prediction),
                    "latin_ratio": ratios["latin"],
                    "devanagari_ratio": ratios["devanagari"],
                }
            )

    predictions = pd.DataFrame(rows)
    runtime = {
        "system": system_name,
        "direction": direction,
        "model_ref": model_ref,
        "device": hardware.device,
        "batch_size": batch_size,
        "sentences": len(predictions),
        "total_inference_seconds": round(float(predictions["latency_seconds"].sum()), 6),
        "average_latency_seconds": round(float(predictions["latency_seconds"].mean()), 6),
        "median_latency_seconds": round(float(predictions["latency_seconds"].median()), 6),
        "p95_latency_seconds": round(float(predictions["latency_seconds"].quantile(0.95)), 6),
        "minimum_latency_seconds": round(float(predictions["latency_seconds"].min()), 6),
        "maximum_latency_seconds": round(float(predictions["latency_seconds"].max()), 6),
        "sentences_per_second": round(
            len(predictions) / max(float(predictions["latency_seconds"].sum()), 1e-9), 4
        ),
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated() / (1024**3), 4)
            if hardware.device == "cuda"
            else None
        ),
    }

    del model
    if hardware.device == "cuda":
        torch.cuda.empty_cache()
    return predictions, runtime


def compute_corpus_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    try:
        from sacrebleu.metrics import BLEU, CHRF, TER
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required for evaluation.") from exc

    references_nested = [list(references)]
    bleu_metric = BLEU(tokenize="13a", effective_order=False)
    chrf_metric = CHRF(word_order=0)
    chrfpp_metric = CHRF(word_order=2)
    ter_metric = TER(normalized=True)

    bleu = bleu_metric.corpus_score(list(predictions), references_nested)
    chrf = chrf_metric.corpus_score(list(predictions), references_nested)
    chrfpp = chrfpp_metric.corpus_score(list(predictions), references_nested)
    ter = ter_metric.corpus_score(list(predictions), references_nested)
    return {
        "sacrebleu": round(float(bleu.score), 4),
        "sacrebleu_signature": str(bleu_metric.get_signature()),
        "chrf": round(float(chrf.score), 4),
        "chrf_signature": str(chrf_metric.get_signature()),
        "chrf_plus_plus": round(float(chrfpp.score), 4),
        "chrf_plus_plus_signature": str(chrfpp_metric.get_signature()),
        "ter": round(float(ter.score), 4),
        "ter_signature": str(ter_metric.get_signature()),
    }


def add_sentence_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        from sacrebleu.metrics import BLEU, CHRF, TER
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required for evaluation.") from exc

    bleu = BLEU(tokenize="13a", effective_order=True)
    chrf = CHRF(word_order=0)
    ter = TER(normalized=True)
    result = frame.copy()
    result["sentence_bleu"] = [
        round(float(bleu.sentence_score(pred, [ref]).score), 4)
        for pred, ref in zip(result["predicted_translation"], result["reference_translation"])
    ]
    result["sentence_chrf"] = [
        round(float(chrf.sentence_score(pred, [ref]).score), 4)
        for pred, ref in zip(result["predicted_translation"], result["reference_translation"])
    ]
    result["sentence_ter"] = [
        round(float(ter.sentence_score(pred, [ref]).score), 4)
        for pred, ref in zip(result["predicted_translation"], result["reference_translation"])
    ]
    return result


def bootstrap_confidence_intervals(
    frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    predictions = frame["predicted_translation"].astype(str).to_numpy()
    references = frame["reference_translation"].astype(str).to_numpy()
    n = len(frame)
    bleu_scores: list[float] = []
    chrf_scores: list[float] = []
    from sacrebleu.metrics import BLEU, CHRF

    bleu_metric = BLEU(tokenize="13a", effective_order=False)
    chrf_metric = CHRF(word_order=0)
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        sampled_predictions = predictions[indices].tolist()
        sampled_references = references[indices].tolist()
        bleu_scores.append(
            float(bleu_metric.corpus_score(sampled_predictions, [sampled_references]).score)
        )
        chrf_scores.append(
            float(chrf_metric.corpus_score(sampled_predictions, [sampled_references]).score)
        )

    def summarize(values: list[float]) -> dict[str, float]:
        lower, upper = np.percentile(values, [2.5, 97.5])
        return {
            "mean": round(float(np.mean(values)), 4),
            "lower_95": round(float(lower), 4),
            "upper_95": round(float(upper), 4),
        }

    return {
        "method": "paired sentence bootstrap resampling within one system",
        "samples": samples,
        "seed": seed,
        "sacrebleu": summarize(bleu_scores),
        "chrf": summarize(chrf_scores),
    }


def summarize_predictions(
    frame: pd.DataFrame,
    runtime: dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    enriched = add_sentence_metrics(frame)
    corpus = compute_corpus_metrics(
        enriched["predicted_translation"].astype(str).tolist(),
        enriched["reference_translation"].astype(str).tolist(),
    )
    direction = str(enriched["direction"].iloc[0])
    expected_script_column = "devanagari_ratio" if direction == "en_hi" else "latin_ratio"
    diagnostics = {
        "exact_match_rate": round(
            float(
                (
                    enriched["predicted_translation"].str.strip()
                    == enriched["reference_translation"].str.strip()
                ).mean()
            ),
            6,
        ),
        "number_preservation_rate": round(float(enriched["numbers_match"].mean()), 6),
        "empty_output_rate": round(
            float((enriched["predicted_translation"].str.strip() == "").mean()), 6
        ),
        "average_expected_script_ratio": round(
            float(enriched[expected_script_column].mean()), 6
        ),
        "average_prediction_reference_length_ratio": round(
            float(
                (
                    enriched["prediction_characters"]
                    / enriched["reference_characters"].clip(lower=1)
                ).mean()
            ),
            6,
        ),
    }
    confidence_intervals = bootstrap_confidence_intervals(
        enriched,
        samples=bootstrap_samples,
        seed=seed,
    )
    summary = {
        "status": "evaluated",
        "system": str(enriched["system"].iloc[0]),
        "direction": direction,
        "model_ref": str(enriched["model_ref"].iloc[0]),
        "examples": int(len(enriched)),
        "metrics": corpus,
        "runtime": runtime,
        "diagnostics": diagnostics,
        "bootstrap_confidence_intervals": confidence_intervals,
    }
    return enriched, summary


def evaluate_system(
    test_frame: pd.DataFrame,
    *,
    system_name: str,
    model_refs: dict[str, str],
    hardware: HardwareProfile,
    config: dict[str, Any],
) -> dict[str, Any]:
    root = Path(config["project_root"])
    output_dir = root / config["outputs"]["root"] / system_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}

    for direction, model_ref in model_refs.items():
        raw, runtime = generate_predictions(
            test_frame,
            model_ref=model_ref,
            direction=direction,
            hardware=hardware,
            config=config,
            system_name=system_name,
        )
        enriched, summary = summarize_predictions(
            raw,
            runtime,
            bootstrap_samples=int(config["profile"]["bootstrap_samples"]),
            seed=int(config["seed"]),
        )
        enriched.to_csv(
            output_dir / f"predictions_{direction}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _write_json(output_dir / f"metrics_{direction}.json", summary)
        summaries[direction] = summary

    _write_json(output_dir / "metrics_combined.json", summaries)
    return summaries


def _trainer_compute_metrics(tokenizer: Any):
    def compute(eval_prediction: Any) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        metrics = compute_corpus_metrics(decoded_predictions, decoded_labels)
        return {
            "sacrebleu": metrics["sacrebleu"],
            "chrf": metrics["chrf"],
        }

    return compute


def fine_tune_direction(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    direction: str,
    hardware: HardwareProfile,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install requirements-evaluation.txt before fine-tuning.") from exc

    source_column, target_column = DIRECTIONS[direction]
    model_id = config["models"][direction]
    root = Path(config["project_root"])
    model_output = root / config["outputs"]["model_root"] / f"fine_tuned_{direction}"
    training_output = root / config["outputs"]["root"] / "training" / direction
    training_output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    max_source = int(config["preprocessing"]["max_source_length"])
    max_target = int(config["preprocessing"]["max_target_length"])

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch[source_column],
            max_length=max_source,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[target_column],
            max_length=max_target,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_pandas(
        train_frame[[source_column, target_column]], preserve_index=False
    ).map(tokenize_batch, batched=True, remove_columns=[source_column, target_column])
    validation_ds = Dataset.from_pandas(
        validation_frame[[source_column, target_column]], preserve_index=False
    ).map(tokenize_batch, batched=True, remove_columns=[source_column, target_column])

    use_bf16 = hardware.device == "cuda" and hardware.bf16_supported
    use_fp16 = hardware.device == "cuda" and not use_bf16
    training_cfg = config["training"]
    profile = config["profile"]

    args = Seq2SeqTrainingArguments(
        output_dir=str(training_output),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(training_cfg["logging_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        label_smoothing_factor=float(training_cfg["label_smoothing_factor"]),
        per_device_train_batch_size=hardware.recommended_train_batch_size,
        per_device_eval_batch_size=hardware.recommended_eval_batch_size,
        gradient_accumulation_steps=hardware.gradient_accumulation_steps,
        num_train_epochs=float(profile["epochs"]),
        predict_with_generate=True,
        generation_num_beams=int(training_cfg["generation_num_beams"]),
        generation_max_length=max_target,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=bool(training_cfg["gradient_checkpointing"]),
        dataloader_num_workers=int(training_cfg["dataloader_num_workers"]),
        save_total_limit=int(training_cfg["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_chrf",
        greater_is_better=True,
        report_to="none",
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        auto_find_batch_size=bool(training_cfg.get("auto_find_batch_size", False)),
    )

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=_trainer_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(training_cfg["early_stopping_patience"])
            )
        ],
    )
    try:
        trainer = Seq2SeqTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Seq2SeqTrainer(tokenizer=tokenizer, **trainer_kwargs)

    started = time.perf_counter()
    train_result = trainer.train()
    duration = time.perf_counter() - started
    trainer.model.config.use_cache = True
    trainer.save_model(str(model_output))
    tokenizer.save_pretrained(str(model_output))

    history = trainer.state.log_history
    history_path = training_output / "training_history.json"
    _write_json(history_path, history)
    summary = {
        "status": "trained",
        "direction": direction,
        "base_model": model_id,
        "output_dir": str(model_output.relative_to(root)),
        "train_examples": len(train_ds),
        "validation_examples": len(validation_ds),
        "epochs_requested": profile["epochs"],
        "training_seconds": round(duration, 3),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
        "hardware": hardware.__dict__,
        "mixed_precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        "training_history": str(history_path.relative_to(root)),
    }
    _write_json(training_output / "training_summary.json", summary)

    del trainer, model
    if hardware.device == "cuda":
        torch.cuda.empty_cache()
    return summary


def fine_tune_both_directions(
    frames: dict[str, pd.DataFrame],
    hardware: HardwareProfile,
    config: dict[str, Any],
) -> dict[str, Any]:
    summaries = {}
    for direction in ["en_hi", "hi_en"]:
        summaries[direction] = fine_tune_direction(
            frames["train"],
            frames["validation"],
            direction=direction,
            hardware=hardware,
            config=config,
        )
    return summaries


def fine_tuned_model_refs(config: dict[str, Any]) -> dict[str, str]:
    root = Path(config["project_root"])
    refs = {
        direction: str(root / config["outputs"]["model_root"] / f"fine_tuned_{direction}")
        for direction in DIRECTIONS
    }
    missing = [path for path in refs.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Fine-tuned model folders are missing: {missing}")
    return refs


def paired_bootstrap_difference(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    from sacrebleu.metrics import BLEU, CHRF

    keys = ["source_text", "reference_translation"]
    merged = baseline[keys + ["predicted_translation"]].merge(
        candidate[keys + ["predicted_translation"]],
        on=keys,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    n = len(merged)
    refs = merged["reference_translation"].astype(str).to_numpy()
    base = merged["predicted_translation_baseline"].astype(str).to_numpy()
    cand = merged["predicted_translation_candidate"].astype(str).to_numpy()
    metrics = {
        "sacrebleu": BLEU(tokenize="13a", effective_order=False),
        "chrf": CHRF(word_order=0),
    }
    output: dict[str, Any] = {
        "method": "paired bootstrap resampling",
        "samples": samples,
        "seed": seed,
        "examples": n,
    }
    for name, metric in metrics.items():
        deltas = []
        for _ in range(samples):
            indices = rng.integers(0, n, size=n)
            sample_refs = refs[indices].tolist()
            base_score = metric.corpus_score(base[indices].tolist(), [sample_refs]).score
            cand_score = metric.corpus_score(cand[indices].tolist(), [sample_refs]).score
            deltas.append(float(cand_score - base_score))
        lower, upper = np.percentile(deltas, [2.5, 97.5])
        output[name] = {
            "mean_delta": round(float(np.mean(deltas)), 4),
            "lower_95": round(float(lower), 4),
            "upper_95": round(float(upper), 4),
            "probability_candidate_better": round(float(np.mean(np.array(deltas) > 0)), 6),
        }
    return output


def _load_system_outputs(config: dict[str, Any], system: str, direction: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(config["project_root"])
    folder = root / config["outputs"]["root"] / system
    predictions = pd.read_csv(folder / f"predictions_{direction}.csv")
    metrics = json.loads((folder / f"metrics_{direction}.json").read_text(encoding="utf-8"))
    return predictions, metrics


def compare_systems(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    significance: dict[str, Any] = {}
    for direction in DIRECTIONS:
        baseline_frame, baseline_metrics = _load_system_outputs(config, "pretrained", direction)
        candidate_frame, candidate_metrics = _load_system_outputs(config, "fine_tuned", direction)
        significance[direction] = paired_bootstrap_difference(
            baseline_frame,
            candidate_frame,
            samples=int(config["profile"]["bootstrap_samples"]),
            seed=int(config["seed"]),
        )
        for system, metrics in [
            ("pretrained", baseline_metrics),
            ("fine_tuned", candidate_metrics),
        ]:
            rows.append(
                {
                    "system": system,
                    "direction": direction,
                    "model_ref": metrics["model_ref"],
                    "examples": metrics["examples"],
                    "sacrebleu": metrics["metrics"]["sacrebleu"],
                    "chrf": metrics["metrics"]["chrf"],
                    "chrf_plus_plus": metrics["metrics"]["chrf_plus_plus"],
                    "ter": metrics["metrics"]["ter"],
                    "average_latency_seconds": metrics["runtime"]["average_latency_seconds"],
                    "p95_latency_seconds": metrics["runtime"]["p95_latency_seconds"],
                    "sentences_per_second": metrics["runtime"]["sentences_per_second"],
                    "number_preservation_rate": metrics["diagnostics"]["number_preservation_rate"],
                }
            )

    comparison = pd.DataFrame(rows)
    summary = {
        "status": "evaluated",
        "profile": config["active_profile"],
        "dataset": config["dataset"]["id"],
        "test_pairs_per_direction": int(comparison["examples"].min()),
        "comparison": comparison.to_dict(orient="records"),
        "paired_bootstrap_significance": significance,
    }
    return comparison, summary


def create_manual_review_candidates(
    config: dict[str, Any],
    *,
    system: str = "fine_tuned",
) -> pd.DataFrame:
    frames = []
    for direction in DIRECTIONS:
        frame, _ = _load_system_outputs(config, system, direction)
        frame = frame.copy()
        frame["heuristic_error_category"] = ""
        frame.loc[frame["predicted_translation"].fillna("").str.strip() == "", "heuristic_error_category"] = "empty_output"
        frame.loc[~frame["numbers_match"].astype(bool), "heuristic_error_category"] = "number_mismatch"
        length_ratio = frame["prediction_characters"] / frame["reference_characters"].clip(lower=1)
        frame.loc[length_ratio < 0.55, "heuristic_error_category"] = "possible_under_translation"
        frame.loc[length_ratio > 1.8, "heuristic_error_category"] = "possible_over_translation"
        repeated = frame["predicted_translation"].fillna("").map(lambda text: bool(REPEATED_TOKEN_RE.search(str(text))))
        frame.loc[repeated, "heuristic_error_category"] = "repetition"
        expected_ratio = frame["devanagari_ratio"] if direction == "en_hi" else frame["latin_ratio"]
        frame.loc[expected_ratio < 0.50, "heuristic_error_category"] = "possible_script_or_copy_error"
        frame.loc[frame["heuristic_error_category"] == "", "heuristic_error_category"] = "low_metric_review"
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["sentence_chrf", "sentence_ter"], ascending=[True, False])
    count = int(config["profile"]["manual_review_examples"])
    selected = combined.head(count).copy()
    selected["human_error_category"] = ""
    selected["human_severity"] = ""
    selected["human_notes"] = ""
    selected["human_translation_quality"] = ""
    columns = [
        "direction",
        "source_text",
        "reference_translation",
        "predicted_translation",
        "sentence_bleu",
        "sentence_chrf",
        "sentence_ter",
        "heuristic_error_category",
        "human_error_category",
        "human_severity",
        "human_translation_quality",
        "human_notes",
    ]
    return selected[columns]


def summarize_manual_review(review: pd.DataFrame) -> dict[str, Any]:
    completed = review[
        review["human_error_category"].fillna("").astype(str).str.strip() != ""
    ].copy()
    if completed.empty:
        return {
            "status": "awaiting_human_review",
            "reviewed_examples": 0,
            "message": "Fill human_error_category, human_severity, human_translation_quality, and human_notes, then rerun this step.",
        }
    return {
        "status": "completed",
        "reviewed_examples": int(len(completed)),
        "error_category_counts": completed["human_error_category"].value_counts().to_dict(),
        "severity_counts": completed["human_severity"].value_counts().to_dict(),
        "quality_counts": completed["human_translation_quality"].value_counts().to_dict(),
        "mean_sentence_chrf": round(float(completed["sentence_chrf"].mean()), 4),
        "notes_completed": int(
            (completed["human_notes"].fillna("").astype(str).str.strip() != "").sum()
        ),
    }


def save_comparison_artifacts(
    comparison: pd.DataFrame,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    root = Path(config["project_root"])
    output_root = root / config["outputs"]["root"]
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = output_root / "model_comparison.csv"
    summary_path = output_root / "comparison_summary.json"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    _write_json(summary_path, summary)
    return {
        "comparison": str(comparison_path.relative_to(root)),
        "summary": str(summary_path.relative_to(root)),
    }


def _metric_row(comparison: pd.DataFrame, system: str, direction: str) -> dict[str, Any]:
    row = comparison[(comparison["system"] == system) & (comparison["direction"] == direction)]
    if row.empty:
        raise ValueError(f"Missing comparison row for {system}/{direction}")
    return row.iloc[0].to_dict()


def sync_portfolio_outputs(
    comparison: pd.DataFrame,
    comparison_summary: dict[str, Any],
    manual_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    root = Path(config["project_root"])
    outputs_cfg = config["outputs"]
    selected_system = "fine_tuned"
    en_hi = _metric_row(comparison, selected_system, "en_hi")
    hi_en = _metric_row(comparison, selected_system, "hi_en")
    pre_en_hi = _metric_row(comparison, "pretrained", "en_hi")
    pre_hi_en = _metric_row(comparison, "pretrained", "hi_en")

    web_payload = {
        "status": "evaluated",
        "dataset": config["dataset"]["id"],
        "profile": config["active_profile"],
        "evaluation_pairs_per_direction": int(comparison["examples"].min()),
        "selected_system": selected_system,
        "note": "Metrics were generated locally from the documented held-out test subset. Browser inference may differ slightly from Python inference because the Static Space uses quantized ONNX models.",
        "en_hi": {
            "sacrebleu": en_hi["sacrebleu"],
            "chrf": en_hi["chrf"],
            "average_latency_seconds": en_hi["average_latency_seconds"],
            "pretrained_sacrebleu": pre_en_hi["sacrebleu"],
            "pretrained_chrf": pre_en_hi["chrf"],
        },
        "hi_en": {
            "sacrebleu": hi_en["sacrebleu"],
            "chrf": hi_en["chrf"],
            "average_latency_seconds": hi_en["average_latency_seconds"],
            "pretrained_sacrebleu": pre_hi_en["sacrebleu"],
            "pretrained_chrf": pre_hi_en["chrf"],
        },
        "manual_error_analysis": manual_summary,
    }
    web_path = root / outputs_cfg["web_metrics"]
    _write_json(web_path, web_payload)

    sacrebleu_payload = {
        "status": "evaluated",
        "dataset": config["dataset"]["id"],
        "profile": config["active_profile"],
        "pretrained": {
            "en_hi": pre_en_hi["sacrebleu"],
            "hi_en": pre_hi_en["sacrebleu"],
        },
        "fine_tuned": {
            "en_hi": en_hi["sacrebleu"],
            "hi_en": hi_en["sacrebleu"],
        },
    }
    chrf_payload = {
        "status": "evaluated",
        "dataset": config["dataset"]["id"],
        "profile": config["active_profile"],
        "pretrained": {"en_hi": pre_en_hi["chrf"], "hi_en": pre_hi_en["chrf"]},
        "fine_tuned": {"en_hi": en_hi["chrf"], "hi_en": hi_en["chrf"]},
    }
    latency_payload = {
        "status": "evaluated",
        "device_note": "Measured on the local evaluation machine; not the browser Static Space.",
        "pretrained": {
            "en_hi": pre_en_hi["average_latency_seconds"],
            "hi_en": pre_hi_en["average_latency_seconds"],
        },
        "fine_tuned": {
            "en_hi": en_hi["average_latency_seconds"],
            "hi_en": hi_en["average_latency_seconds"],
        },
    }
    model_metrics = {
        "status": "evaluated",
        "dataset": config["dataset"]["id"],
        "profile": config["active_profile"],
        "selected_system": selected_system,
        "comparison": comparison.to_dict(orient="records"),
        "paired_bootstrap_significance": comparison_summary[
            "paired_bootstrap_significance"
        ],
        "manual_error_analysis": manual_summary,
        "deployment_status": {
            "python_fine_tuned_models": "ready locally",
            "static_space_models": "pretrained quantized ONNX until a later export step",
            "onnx_export_pending": True,
        },
    }

    target_payloads = {
        root / outputs_cfg["root_sacrebleu"]: sacrebleu_payload,
        root / outputs_cfg["root_chrf"]: chrf_payload,
        root / outputs_cfg["root_latency"]: latency_payload,
        root / outputs_cfg["root_model_metrics"]: model_metrics,
    }
    for path, payload in target_payloads.items():
        _write_json(path, payload)

    root_comparison_path = root / outputs_cfg["root_comparison"]
    comparison.to_csv(root_comparison_path, index=False, encoding="utf-8-sig")
    return {
        "web_metrics": str(web_path.relative_to(root)),
        "sacrebleu": outputs_cfg["root_sacrebleu"],
        "chrf": outputs_cfg["root_chrf"],
        "latency": outputs_cfg["root_latency"],
        "model_metrics": outputs_cfg["root_model_metrics"],
        "comparison": outputs_cfg["root_comparison"],
    }


def create_plots(comparison: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt

    root = Path(config["project_root"])
    output_dir = root / config["outputs"]["root"] / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for metric in ["sacrebleu", "chrf", "ter", "average_latency_seconds"]:
        pivot = comparison.pivot(index="direction", columns="system", values=metric)
        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_title(f"{metric.replace('_', ' ').title()}: Pretrained vs Fine-tuned")
        ax.set_xlabel("Translation direction")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        path = output_dir / f"comparison_{metric}.png"
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        paths.append(str(path.relative_to(root)))

    for system in ["pretrained", "fine_tuned"]:
        for direction in DIRECTIONS:
            prediction_path = (
                root
                / config["outputs"]["root"]
                / system
                / f"predictions_{direction}.csv"
            )
            if not prediction_path.exists():
                continue
            frame = pd.read_csv(prediction_path)
            plt.figure(figsize=(8, 5))
            plt.scatter(frame["source_tokens"], frame["latency_seconds"], alpha=0.35)
            plt.xlabel("Source tokens")
            plt.ylabel("Latency per sentence (seconds)")
            plt.title(f"Latency by source length: {system} {direction}")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            path = output_dir / f"latency_by_length_{system}_{direction}.png"
            plt.savefig(path, dpi=180, bbox_inches="tight")
            plt.close()
            paths.append(str(path.relative_to(root)))

    for direction in DIRECTIONS:
        history_path = (
            root
            / config["outputs"]["root"]
            / "training"
            / direction
            / "training_history.json"
        )
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        train_points = [item for item in history if "loss" in item and "step" in item]
        eval_points = [item for item in history if "eval_loss" in item and "step" in item]
        if not train_points and not eval_points:
            continue
        plt.figure(figsize=(8, 5))
        if train_points:
            plt.plot(
                [item["step"] for item in train_points],
                [item["loss"] for item in train_points],
                marker="o",
                markersize=3,
                label="training loss",
            )
        if eval_points:
            plt.plot(
                [item["step"] for item in eval_points],
                [item["eval_loss"] for item in eval_points],
                marker="s",
                markersize=4,
                label="validation loss",
            )
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title(f"Training history: {direction}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        path = output_dir / f"training_history_{direction}.png"
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        paths.append(str(path.relative_to(root)))

    return paths


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
