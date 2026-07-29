"""Comprehensive held-out evaluation for base FLAN-T5 and the LoRA adapter.

The module intentionally separates automated quality proxies from human factuality
review. It saves per-example predictions so every aggregate metric is auditable.
"""

from __future__ import annotations

import gc
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_preprocessing import load_jsonl
from .experiment_utils import count_parameters, hardware_info, save_json, set_reproducibility
from .prompt_templates import format_prompt

STOP_WORDS = {
    "about", "after", "again", "against", "also", "and", "answer", "are", "because", "been", "before",
    "between", "build", "can", "compare", "could", "data", "does", "explain", "for", "from", "give", "how",
    "into", "its", "machine", "model", "more", "most", "should", "small", "than", "that", "the", "their",
    "then", "these", "they", "this", "through", "using", "what", "when", "where", "which", "while", "with",
    "would", "your",
}
ABSOLUTE_MARKERS = {"always", "never", "guaranteed", "perfect", "perfectly", "100%", "completely eliminates"}
ATTRIBUTION_PATTERNS = re.compile(r"\b(?:according to|research proves|studies show|experts say)\b", re.IGNORECASE)
REFUSAL_PATTERNS = re.compile(r"\b(?:i cannot|i can't|i am unable|not able to)\b", re.IGNORECASE)


@dataclass(frozen=True)
class EvaluationConfig:
    max_source_length: int = 384
    max_target_length: int = 192
    max_new_tokens: int = 192
    num_beams: int = 4
    repetition_penalty: float = 1.05
    batch_size_loss: int = 8
    seed: int = 42
    bertscore_model_type: str = "distilbert-base-uncased"
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    bootstrap_samples: int = 5000
    confidence_level: float = 0.95
    low_reference_support_threshold: float = 0.42


def _device_and_dtype():
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            precision = "bf16"
        else:
            dtype = torch.float16
            precision = "fp16"
    else:
        device = "cpu"
        dtype = torch.float32
        precision = "fp32"
    return device, dtype, precision


def load_seq2seq_model(base_model_id: str, adapter_path: str | Path | None = None):
    """Load either the base model or a compatible PEFT adapter for evaluation."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device, dtype, precision = _device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    kwargs: dict[str, Any] = {}
    if device == "cuda":
        kwargs["torch_dtype"] = dtype
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, **kwargs)
    mode = "base_model"
    adapter_id = None
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        mode = "lora_adapter"
        adapter_id = str(adapter_path)
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "precision": precision,
        "base_model_id": base_model_id,
        "adapter_path": adapter_id,
        "mode": mode,
        "parameters": count_parameters(model),
    }


def unload_model(bundle: dict[str, Any] | None) -> None:
    if bundle:
        bundle.pop("model", None)
        bundle.pop("tokenizer", None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _synchronize(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def _generate_one(bundle: dict[str, Any], prompt: str, config: EvaluationConfig) -> tuple[str, float, int, int, float | None]:
    import torch

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_source_length)
    input_tokens = int(encoded["input_ids"].shape[-1])
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            do_sample=False,
            repetition_penalty=config.repetition_penalty,
            early_stopping=True,
        )
    _synchronize(device)
    latency = time.perf_counter() - start
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    output_tokens = int(output_ids.shape[-1])
    peak_memory = None
    if device == "cuda":
        peak_memory = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
    return text, latency, input_tokens, output_tokens, peak_memory


def generate_predictions(bundle: dict[str, Any], rows: list[dict[str, Any]], config: EvaluationConfig) -> pd.DataFrame:
    """Generate deterministic predictions and record per-example systems metrics."""
    if rows:
        warmup_prompt = format_prompt(rows[0]["instruction"], rows[0].get("input", ""))
        _generate_one(bundle, warmup_prompt, config)
    records: list[dict[str, Any]] = []
    for row in rows:
        prompt = format_prompt(row["instruction"], row.get("input", ""))
        generated, latency, input_tokens, output_tokens, peak_memory = _generate_one(bundle, prompt, config)
        records.append({
            **row,
            "prompt": prompt,
            "generated_answer": generated,
            "model": bundle["mode"],
            "base_model_id": bundle["base_model_id"],
            "adapter_path": bundle["adapter_path"],
            "device": bundle["device"],
            "precision": bundle["precision"],
            "latency_seconds": round(latency, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": round(output_tokens / latency, 4) if latency > 0 else None,
            "peak_gpu_memory_gb": peak_memory,
        })
    return pd.DataFrame(records)


def compute_heldout_loss(bundle: dict[str, Any], rows: list[dict[str, Any]], config: EvaluationConfig) -> dict[str, float | None]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    features = []
    for row in rows:
        source = tokenizer(
            format_prompt(row["instruction"], row.get("input", "")),
            truncation=True,
            max_length=config.max_source_length,
        )
        target = tokenizer(text_target=row["reference_answer"], truncation=True, max_length=config.max_target_length)
        source["labels"] = target["input_ids"]
        features.append(source)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    loader = DataLoader(features, batch_size=config.batch_size_loss, shuffle=False, collate_fn=collator)
    weighted_loss = 0.0
    example_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_size = int(batch["input_ids"].shape[0])
            weighted_loss += float(outputs.loss.detach().float().cpu()) * batch_size
            example_count += batch_size
    mean_loss = weighted_loss / example_count if example_count else None
    perplexity = math.exp(mean_loss) if mean_loss is not None and mean_loss < 20 else None
    return {"heldout_loss": mean_loss, "perplexity": perplexity}


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]{2,}", text)
        if token.lower() not in STOP_WORDS
    }


def _entities_for_comparison(instruction: str) -> list[str]:
    match = re.search(r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)[?.]?$", instruction.strip(), flags=re.IGNORECASE)
    if not match:
        return []
    return [match.group(1).strip().lower(), match.group(2).strip().lower()]


def score_instruction_adherence(instruction: str, response: str, category: str) -> dict[str, Any]:
    lower = response.lower().strip()
    words = re.findall(r"\b\w+\b", response)
    answered = len(words) >= 15
    no_unnecessary_refusal = not bool(REFUSAL_PATTERNS.search(lower))
    requested_terms = _terms(instruction)
    response_terms = _terms(response)
    topic_coverage = len(requested_terms & response_terms) / max(len(requested_terms), 1)
    topic_coverage = min(topic_coverage * 1.8, 1.0)

    format_score = 1.0
    category = (category or "").lower()
    if category == "code_example" or "python example" in instruction.lower() or "code" in instruction.lower():
        format_score = float("```" in response and any(marker in lower for marker in ("import ", "def ", "from ", "=")))
    elif category == "algorithm_comparison" or instruction.lower().startswith("compare"):
        entities = _entities_for_comparison(instruction)
        entity_hits = sum(float(entity in lower or all(part in lower for part in _terms(entity))) for entity in entities)
        contrast = float(any(marker in lower for marker in ("while", "whereas", "however", "difference", "use ", "choose")))
        format_score = (entity_hits / max(len(entities), 1)) * 0.7 + contrast * 0.3
    elif category == "workflow_explanation":
        format_score = float(bool(re.search(r"(?:^|\s)(?:1[.)]|first|start|define|step)", lower)))
    elif category == "metric_explanation":
        format_score = float(any(marker in lower for marker in ("measures", "fraction", "average", "formula", "use it", "limitation")))

    length_score = 1.0 if 20 <= len(words) <= 220 else (0.6 if 10 <= len(words) <= 300 else 0.2)
    repetition_score = 1.0
    trigrams = [tuple(words[i : i + 3]) for i in range(max(len(words) - 2, 0))]
    if trigrams:
        repetition_ratio = 1 - len(set(trigrams)) / len(trigrams)
        repetition_score = max(0.0, 1.0 - repetition_ratio * 3)
    score = (
        0.25 * float(answered)
        + 0.25 * topic_coverage
        + 0.25 * format_score
        + 0.10 * float(no_unnecessary_refusal)
        + 0.10 * length_score
        + 0.05 * repetition_score
    )
    return {
        "adherence_score": round(float(score), 6),
        "adherence_answered": bool(answered),
        "adherence_topic_coverage": round(float(topic_coverage), 6),
        "adherence_format_score": round(float(format_score), 6),
        "adherence_no_unnecessary_refusal": bool(no_unnecessary_refusal),
        "adherence_length_score": round(float(length_score), 6),
        "adherence_repetition_score": round(float(repetition_score), 6),
    }


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", text))


def score_hallucination_risk(instruction: str, response: str, reference: str, reference_support: float, config: EvaluationConfig) -> dict[str, Any]:
    lower = response.lower()
    context_numbers = _numbers(instruction + " " + reference)
    generated_numbers = _numbers(response)
    unsupported_numbers = sorted(generated_numbers - context_numbers)
    flags: list[str] = []
    if any(marker in lower for marker in ABSOLUTE_MARKERS):
        flags.append("overconfident_absolute_language")
    if ATTRIBUTION_PATTERNS.search(response) and not re.search(r"https?://|\[[0-9]+\]", response):
        flags.append("unsupported_attribution")
    if unsupported_numbers:
        flags.append("unsupported_numeric_claim_review")
    if reference_support < config.low_reference_support_threshold:
        flags.append("low_reference_support")
    if not response.strip():
        flags.append("empty_response")
    severity = "none"
    if flags:
        severity = "high" if "empty_response" in flags else ("medium" if len(flags) >= 2 else "review")
    return {
        "hallucination_risk_flag": bool(flags),
        "hallucination_risk_types": flags,
        "hallucination_risk_severity": severity,
        "unsupported_numbers": unsupported_numbers,
        "reference_support_score": round(float(reference_support), 6),
    }


def add_automatic_metrics(frame: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    """Add ROUGE, BERTScore, semantic similarity, adherence, and risk flags."""
    if frame.empty:
        return frame
    predictions = frame["generated_answer"].fillna("").astype(str).tolist()
    references = frame["reference_answer"].fillna("").astype(str).tolist()
    instructions = frame["instruction"].fillna("").astype(str).tolist()

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_rows = [scorer.score(reference, prediction) for reference, prediction in zip(references, predictions)]
    frame["rouge1"] = [row["rouge1"].fmeasure for row in rouge_rows]
    frame["rouge2"] = [row["rouge2"].fmeasure for row in rouge_rows]
    frame["rougeL"] = [row["rougeL"].fmeasure for row in rouge_rows]

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config.embedding_model_id)
    reference_embeddings = embedder.encode(references, normalize_embeddings=True, show_progress_bar=False)
    prediction_embeddings = embedder.encode(predictions, normalize_embeddings=True, show_progress_bar=False)
    instruction_embeddings = embedder.encode(instructions, normalize_embeddings=True, show_progress_bar=False)
    reference_similarity = np.sum(reference_embeddings * prediction_embeddings, axis=1)
    prompt_similarity = np.sum(instruction_embeddings * prediction_embeddings, axis=1)
    frame["semantic_reference_similarity"] = reference_similarity
    frame["semantic_prompt_similarity"] = prompt_similarity
    frame["semantic_relevance"] = 0.75 * reference_similarity + 0.25 * prompt_similarity
    del embedder, reference_embeddings, prediction_embeddings, instruction_embeddings
    gc.collect()

    from bert_score import score as bert_score

    bert_device = "cuda" if hardware_info().get("cuda_available") else "cpu"
    precision, recall, f1 = bert_score(
        predictions,
        references,
        model_type=config.bertscore_model_type,
        device=bert_device,
        verbose=False,
        rescale_with_baseline=False,
    )
    frame["bertscore_precision"] = precision.detach().float().cpu().numpy()
    frame["bertscore_recall"] = recall.detach().float().cpu().numpy()
    frame["bertscore_f1"] = f1.detach().float().cpu().numpy()

    adherence_rows = [
        score_instruction_adherence(instruction, response, category)
        for instruction, response, category in zip(frame["instruction"], frame["generated_answer"], frame["category"])
    ]
    for key in adherence_rows[0]:
        frame[key] = [row[key] for row in adherence_rows]

    risk_rows = [
        score_hallucination_risk(instruction, response, reference, support, config)
        for instruction, response, reference, support in zip(
            frame["instruction"], frame["generated_answer"], frame["reference_answer"], frame["semantic_reference_similarity"]
        )
    ]
    for key in risk_rows[0]:
        frame[key] = [row[key] for row in risk_rows]
    return frame


def summarize_model(frame: pd.DataFrame, loss_metrics: dict[str, Any], bundle_metadata: dict[str, Any], config: EvaluationConfig) -> dict[str, Any]:
    metrics = [
        "bertscore_precision", "bertscore_recall", "bertscore_f1", "rouge1", "rouge2", "rougeL",
        "semantic_reference_similarity", "semantic_prompt_similarity", "semantic_relevance", "adherence_score",
        "reference_support_score", "latency_seconds", "tokens_per_second", "output_tokens",
    ]
    summary: dict[str, Any] = {
        "status": "completed",
        "model": bundle_metadata["mode"],
        "base_model_id": bundle_metadata["base_model_id"],
        "adapter_path": bundle_metadata.get("adapter_path"),
        "device": bundle_metadata["device"],
        "precision": bundle_metadata["precision"],
        "evaluated_examples": int(len(frame)),
        "parameters": bundle_metadata["parameters"],
        "evaluation_config": asdict(config),
        "heldout_loss": loss_metrics.get("heldout_loss"),
        "perplexity": loss_metrics.get("perplexity"),
        "hallucination_risk_flag_rate": float(frame["hallucination_risk_flag"].mean()),
        "note": "Automated metrics are quality proxies. Human factuality review remains required.",
    }
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce")
        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"median_{metric}"] = float(values.median())
        summary[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if "peak_gpu_memory_gb" in frame and frame["peak_gpu_memory_gb"].notna().any():
        summary["max_peak_gpu_memory_gb"] = float(pd.to_numeric(frame["peak_gpu_memory_gb"], errors="coerce").max())
    return summary


def category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["bertscore_f1", "rougeL", "semantic_relevance", "adherence_score", "reference_support_score", "latency_seconds"]
    return (
        frame.groupby(["model", "category"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .round(6)
        .reset_index()
    )


def _bootstrap_paired(delta: np.ndarray, samples: int, confidence_level: float, seed: int) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if not len(delta):
        return {"mean_delta": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    boot_means = delta[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci_low": float(np.quantile(boot_means, alpha)),
        "ci_high": float(np.quantile(boot_means, 1 - alpha)),
        "n": int(len(delta)),
        "win_rate": float(np.mean(delta > 0)),
        "tie_rate": float(np.mean(np.isclose(delta, 0))),
        "loss_rate": float(np.mean(delta < 0)),
    }


def compare_models(base_frame: pd.DataFrame, lora_frame: pd.DataFrame, config: EvaluationConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    key_columns = ["id", "instruction", "category", "topic", "reference_answer"]
    base_columns = key_columns + [
        "generated_answer", "bertscore_f1", "rougeL", "semantic_relevance", "adherence_score",
        "reference_support_score", "hallucination_risk_flag", "latency_seconds", "tokens_per_second",
    ]
    lora_columns = base_columns
    merged = base_frame[base_columns].merge(lora_frame[lora_columns], on=key_columns, suffixes=("_base", "_lora"), validate="one_to_one")
    higher_is_better = ["bertscore_f1", "rougeL", "semantic_relevance", "adherence_score", "reference_support_score", "tokens_per_second"]
    lower_is_better = ["latency_seconds"]
    comparison: dict[str, Any] = {"status": "completed", "paired_examples": int(len(merged)), "metrics": {}}
    for metric in higher_is_better:
        merged[f"delta_{metric}"] = merged[f"{metric}_lora"] - merged[f"{metric}_base"]
        comparison["metrics"][metric] = _bootstrap_paired(
            merged[f"delta_{metric}"].to_numpy(), config.bootstrap_samples, config.confidence_level, config.seed
        )
    for metric in lower_is_better:
        merged[f"delta_{metric}"] = merged[f"{metric}_base"] - merged[f"{metric}_lora"]
        stats = _bootstrap_paired(merged[f"delta_{metric}"].to_numpy(), config.bootstrap_samples, config.confidence_level, config.seed)
        stats["delta_definition"] = "positive means the LoRA model is faster"
        comparison["metrics"][metric] = stats
    merged["hallucination_risk_improved"] = (
        merged["hallucination_risk_flag_base"].astype(int) - merged["hallucination_risk_flag_lora"].astype(int)
    )
    comparison["hallucination_risk_flag_rate_base"] = float(merged["hallucination_risk_flag_base"].mean())
    comparison["hallucination_risk_flag_rate_lora"] = float(merged["hallucination_risk_flag_lora"].mean())
    comparison["note"] = "Paired bootstrap intervals quantify uncertainty on this held-out set; they do not establish universal superiority."
    try:
        from scipy.stats import wilcoxon

        for metric in higher_is_better:
            delta = merged[f"delta_{metric}"].to_numpy(dtype=float)
            if np.any(~np.isclose(delta, 0)):
                result = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
                comparison["metrics"][metric]["wilcoxon_p_value"] = float(result.pvalue)
            else:
                comparison["metrics"][metric]["wilcoxon_p_value"] = None
    except Exception as exc:
        comparison["wilcoxon_note"] = f"Wilcoxon test not available: {exc}"
    return merged, comparison


def save_manual_review_template(comparison_frame: pd.DataFrame, path: str | Path, sample_size: int = 24, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    groups = []
    per_category = max(1, sample_size // max(comparison_frame["category"].nunique(), 1))
    for _, group in comparison_frame.groupby("category"):
        take = min(per_category, len(group))
        groups.append(group.iloc[rng.choice(len(group), size=take, replace=False)])
    selected = pd.concat(groups, ignore_index=True) if groups else comparison_frame.head(sample_size)
    if len(selected) < min(sample_size, len(comparison_frame)):
        remaining = comparison_frame[~comparison_frame["id"].isin(selected["id"])]
        take = min(sample_size - len(selected), len(remaining))
        if take:
            selected = pd.concat([selected, remaining.iloc[rng.choice(len(remaining), size=take, replace=False)]], ignore_index=True)
    output = selected[[
        "id", "category", "topic", "instruction", "reference_answer", "generated_answer_base", "generated_answer_lora"
    ]].copy()
    for column in [
        "base_correctness_1_to_5", "lora_correctness_1_to_5", "base_relevance_1_to_5", "lora_relevance_1_to_5",
        "base_clarity_1_to_5", "lora_clarity_1_to_5", "preferred_model", "hallucination_present_base",
        "hallucination_present_lora", "reviewer_notes",
    ]:
        output[column] = ""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)


def save_before_after_markdown(comparison_frame: pd.DataFrame, path: str | Path, n: int = 12) -> None:
    ranked = comparison_frame.sort_values("delta_bertscore_f1", ascending=False).head(n)
    lines = ["# Before vs After LoRA Fine-Tuning", "", "These examples are selected by held-out BERTScore improvement. Review them manually before making qualitative claims.", ""]
    for _, row in ranked.iterrows():
        lines.extend([
            f"## {row['topic']} — `{row['id']}`",
            "",
            f"**Instruction:** {row['instruction']}",
            "",
            f"**Reference:** {row['reference_answer']}",
            "",
            "**Base FLAN-T5 response:**",
            "",
            row["generated_answer_base"],
            "",
            "**LoRA response:**",
            "",
            row["generated_answer_lora"],
            "",
            f"**BERTScore F1 change:** {row['delta_bertscore_f1']:.4f}",
            "",
            "---",
            "",
        ])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def save_hallucination_report(base_frame: pd.DataFrame, lora_frame: pd.DataFrame, path: str | Path) -> None:
    lines = [
        "# Automated Hallucination-Risk Triage",
        "",
        "This report flags low reference support, unsupported numeric claims, unsupported attributions, and overconfident language. It is not a factuality verdict; complete the manual review template.",
        "",
    ]
    for label, frame in (("Base model", base_frame), ("LoRA adapter", lora_frame)):
        flagged = frame[frame["hallucination_risk_flag"]].copy()
        lines.extend([f"## {label}", "", f"Flagged examples: **{len(flagged)} / {len(frame)}**", ""])
        for _, row in flagged.head(20).iterrows():
            lines.extend([
                f"### {row['id']} — {row['topic']}",
                f"- Risk types: `{row['hallucination_risk_types']}`",
                f"- Reference support: `{row['reference_support_score']:.4f}`",
                f"- Generated response: {row['generated_answer']}",
                "",
            ])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def plot_comparison(base_summary: dict[str, Any], lora_summary: dict[str, Any], comparison_frame: pd.DataFrame, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = ["bertscore_f1", "rougeL", "semantic_relevance", "adherence_score", "reference_support_score"]
    labels = ["BERTScore F1", "ROUGE-L", "Semantic relevance", "Instruction adherence", "Reference support"]
    base_values = [base_summary[f"mean_{metric}"] for metric in metrics]
    lora_values = [lora_summary[f"mean_{metric}"] for metric in metrics]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, base_values, width, label="Base FLAN-T5")
    ax.bar(x + width / 2, lora_values, width, label="LoRA adapter")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title("Held-Out Base vs LoRA Evaluation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "base_vs_lora_metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    category = comparison_frame.groupby("category")["delta_bertscore_f1"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    category.plot(kind="barh", ax=ax)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Mean paired BERTScore F1 improvement")
    ax.set_title("LoRA Improvement by Prompt Category")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "bertscore_improvement_by_category.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(
        [comparison_frame["latency_seconds_base"], comparison_frame["latency_seconds_lora"]],
        tick_labels=["Base FLAN-T5", "LoRA adapter"],
        showmeans=True,
    )
    ax.set_ylabel("Seconds per response")
    ax.set_title("Warm-Cache Inference Latency")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "latency_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_model_bundle(
    bundle: dict[str, Any],
    rows: list[dict[str, Any]],
    config: EvaluationConfig,
    output_dir: str | Path,
    file_prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = generate_predictions(bundle, rows, config)
    loss_metrics = compute_heldout_loss(bundle, rows, config)
    scored = add_automatic_metrics(predictions, config)
    summary = summarize_model(scored, loss_metrics, bundle, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / f"{file_prefix}_per_example.csv", index=False)
    save_json(summary, output / f"{file_prefix}_metrics.json")
    return scored, summary


def run_base_vs_lora_evaluation(
    base_model_id: str,
    adapter_path: str | Path,
    evaluation_path: str | Path,
    output_dir: str | Path,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    config = config or EvaluationConfig()
    set_reproducibility(config.seed, deterministic=True)
    rows = load_jsonl(evaluation_path)
    if not rows:
        raise ValueError(f"No evaluation rows found in {evaluation_path}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    base_bundle = load_seq2seq_model(base_model_id)
    base_frame, base_summary = evaluate_model_bundle(base_bundle, rows, config, output, "base_model")
    unload_model(base_bundle)

    lora_bundle = load_seq2seq_model(base_model_id, adapter_path)
    lora_frame, lora_summary = evaluate_model_bundle(lora_bundle, rows, config, output, "lora_model")
    unload_model(lora_bundle)

    comparison_frame, comparison_summary = compare_models(base_frame, lora_frame, config)
    comparison_frame.to_csv(output / "base_vs_lora_per_example.csv", index=False)
    save_json(comparison_summary, output / "base_vs_lora_comparison.json")

    combined = pd.concat([base_frame, lora_frame], ignore_index=True)
    combined.to_csv(output / "generated_response_examples.csv", index=False)
    categories = category_summary(combined)
    categories.to_csv(output / "category_metrics.csv", index=False)

    instruction_payload = {
        "status": "completed",
        "base_mean": base_summary["mean_adherence_score"],
        "lora_mean": lora_summary["mean_adherence_score"],
        "paired_delta": comparison_summary["metrics"]["adherence_score"],
        "per_example_file": "base_vs_lora_per_example.csv",
        "note": "Transparent heuristic rubric; complete human review before final claims.",
    }
    bert_payload = {
        "status": "completed",
        "model_type": config.bertscore_model_type,
        "base": {
            "precision": base_summary["mean_bertscore_precision"],
            "recall": base_summary["mean_bertscore_recall"],
            "f1": base_summary["mean_bertscore_f1"],
        },
        "lora": {
            "precision": lora_summary["mean_bertscore_precision"],
            "recall": lora_summary["mean_bertscore_recall"],
            "f1": lora_summary["mean_bertscore_f1"],
        },
        "paired_delta_f1": comparison_summary["metrics"]["bertscore_f1"],
        "note": "BERTScore is semantic similarity, not factual correctness.",
    }
    relevance_payload = {
        "status": "completed",
        "embedding_model": config.embedding_model_id,
        "base_mean": base_summary["mean_semantic_relevance"],
        "lora_mean": lora_summary["mean_semantic_relevance"],
        "paired_delta": comparison_summary["metrics"]["semantic_relevance"],
        "note": "Embedding similarity is a proxy and should be interpreted with examples.",
    }
    model_metrics = {
        "status": "completed",
        "base_model": base_summary,
        "lora_model": lora_summary,
        "comparison": comparison_summary,
        "hardware": hardware_info(),
        "evaluation_dataset": str(evaluation_path),
        "audit_files": {
            "base_predictions": "base_model_per_example.csv",
            "lora_predictions": "lora_model_per_example.csv",
            "paired_predictions": "base_vs_lora_per_example.csv",
            "manual_review": "manual_review_results.csv",
        },
    }
    save_json(instruction_payload, output / "instruction_adherence_results.json")
    save_json(bert_payload, output / "bertscore_results.json")
    save_json(relevance_payload, output / "response_relevance_results.json")
    save_json(model_metrics, output / "model_metrics.json")

    save_manual_review_template(comparison_frame, output / "manual_review_results.csv", seed=config.seed)
    save_before_after_markdown(comparison_frame, output / "before_after_finetuning_examples.md")
    save_hallucination_report(base_frame, lora_frame, output / "hallucination_analysis.md")
    plot_comparison(base_summary, lora_summary, comparison_frame, output)

    # Mirror the principal artifacts into outputs/ so the Gradio app and the
    # original project paths automatically move from status:not_run to real results.
    if output.name == "portfolio_experiment":
        mirror_dir = output.parent
        for filename in [
            "model_metrics.json",
            "bertscore_results.json",
            "instruction_adherence_results.json",
            "response_relevance_results.json",
            "generated_response_examples.csv",
            "manual_review_results.csv",
            "before_after_finetuning_examples.md",
            "hallucination_analysis.md",
            "base_vs_lora_metric_comparison.png",
            "bertscore_improvement_by_category.png",
            "latency_comparison.png",
        ]:
            source = output / filename
            if source.exists():
                shutil.copy2(source, mirror_dir / filename)

    manifest = {
        "status": "completed",
        "base_model_id": base_model_id,
        "adapter_path": str(adapter_path),
        "evaluation_path": str(evaluation_path),
        "examples": len(rows),
        "config": asdict(config),
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    save_json(manifest, output / "evaluation_manifest.json")
    return model_metrics
