"""RTX-assisted synthetic dataset expansion with validation and leakage guards.

A stronger local instruction model acts only as a *teacher* to draft candidate
records. Candidates are validated, de-duplicated, checked against the held-out
benchmark, and saved for human review before student-model training.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config import DatasetGenerationConfig
from .data_preprocessing import load_jsonl, save_jsonl, validate_and_clean_records
from .hardware_utils import HardwareProfile, detect_hardware

ALLOWED_CATEGORIES = {
    "Concept explanation",
    "Algorithm comparison",
    "Metric explanation",
    "Example generation",
    "Beginner-friendly explanation",
    "Interview-style answer",
    "Small code example",
    "Data Science workflow",
    "ML project guidance",
    "Quality analytics",
}
ALLOWED_DIFFICULTIES = {"beginner", "intermediate", "advanced"}


def load_topic_plan(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("The dataset generation plan must be a non-empty JSON list.")
    return [dict(item) for item in data]


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Teacher response was not a JSON array.")
    return [dict(item) for item in parsed if isinstance(item, dict)]


def build_teacher_prompt(topic: Dict[str, Any], examples_per_topic: int) -> str:
    comparison = str(topic.get("compare_to", "")).strip()
    context = str(topic.get("context", "")).strip()
    return f"""Create exactly {examples_per_topic} high-quality instruction-tuning records for an educational Machine Learning and Data Science assistant.

TOPIC: {topic['topic']}
OPTIONAL COMPARISON TOPIC: {comparison or 'none'}
DOMAIN CONTEXT: {context or 'general ML/Data Science and non-confidential quality analytics'}

Return ONLY a valid JSON array. Each object must contain exactly these keys:
- instruction
- input
- output
- category
- difficulty
- topic
- source

Requirements:
1. Use diverse categories chosen from: {sorted(ALLOWED_CATEGORIES)}.
2. Use beginner, intermediate, and advanced difficulty levels where appropriate.
3. Answers must be factually careful, self-contained, educational, and approximately 60-180 words.
4. Include practical examples, trade-offs, and caveats when useful.
5. Code examples must be short, correct Python or clear pseudocode.
6. Do not claim that any algorithm is always best.
7. Do not invent studies, citations, benchmark scores, or numerical facts.
8. Do not include private company data, personal data, legal, medical, financial, immigration, or safety-critical advice.
9. Make every instruction distinct and useful to a learner or interviewer.
10. Set source to "local-teacher-synthetic-reviewed" and topic to "{topic['topic']}".
11. Do not use markdown fences around the JSON.
"""


def _teacher_model_id(config: DatasetGenerationConfig, hardware: HardwareProfile) -> str:
    if config.teacher_model_id:
        if hardware.gpu_vram_gb and hardware.gpu_vram_gb < 8 and "1.5B" in config.teacher_model_id:
            return "Qwen/Qwen2.5-0.5B-Instruct"
        return config.teacher_model_id
    return "Qwen/Qwen2.5-1.5B-Instruct" if hardware.gpu_vram_gb >= 8 else "Qwen/Qwen2.5-0.5B-Instruct"


def load_teacher_model(
    config: DatasetGenerationConfig | None = None,
    hardware: HardwareProfile | None = None,
) -> tuple[Any, Any, str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install requirements-training.txt before generating the enhanced dataset.") from exc

    cfg = config or DatasetGenerationConfig()
    hw = hardware or detect_hardware()
    if not hw.cuda_available:
        raise RuntimeError("CUDA is required for the local teacher-model dataset generation workflow.")

    model_id = _teacher_model_id(cfg, hw)
    dtype = torch.bfloat16 if hw.bf16_supported else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer, model_id


def generate_topic_records(
    model: Any,
    tokenizer: Any,
    topic: Dict[str, Any],
    config: DatasetGenerationConfig,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch

    prompt = build_teacher_prompt(topic, config.examples_per_topic)
    messages = [
        {"role": "system", "content": "You create accurate educational ML/Data Science instruction datasets as strict JSON."},
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = generated[0][encoded["input_ids"].shape[-1] :]
    text = tokenizer.decode(completion, skip_special_tokens=True)
    records = _extract_json_array(text)
    elapsed = time.perf_counter() - started
    return records, {
        "topic": topic["topic"],
        "generated_records": len(records),
        "latency_seconds": round(elapsed, 3),
        "raw_response": text,
    }


def normalize_generated_records(records: Iterable[Dict[str, Any]], topic_name: str, start_id: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for offset, raw in enumerate(records):
        category = str(raw.get("category", "Concept explanation")).strip()
        difficulty = str(raw.get("difficulty", "intermediate")).strip().lower()
        item = {
            "id": f"ml_ds_v2_{start_id + offset:05d}",
            "instruction": str(raw.get("instruction", "")).strip(),
            "input": str(raw.get("input", "")).strip(),
            "output": str(raw.get("output", raw.get("response", ""))).strip(),
            "category": category if category in ALLOWED_CATEGORIES else "Concept explanation",
            "difficulty": difficulty if difficulty in ALLOWED_DIFFICULTIES else "intermediate",
            "topic": topic_name,
            "source": "local-teacher-synthetic-reviewed",
            "split": "train",
        }
        normalized.append(item)
    return normalized


def remove_near_duplicates(
    records: Sequence[Dict[str, Any]], threshold: float = 0.88
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove near-duplicate instructions using word and character TF-IDF similarity."""
    if len(records) < 2:
        return list(records), []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError("scikit-learn is required for dataset de-duplication.") from exc

    texts = [str(r["instruction"]).lower() for r in records]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(matrix)
    keep: List[int] = []
    removed: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        nearest = max((float(similarity[index, prior]) for prior in keep), default=0.0)
        if nearest >= threshold:
            removed.append({"id": record.get("id"), "instruction": record.get("instruction"), "nearest_similarity": round(nearest, 4)})
        else:
            keep.append(index)
    return [dict(records[i]) for i in keep], removed


def remove_benchmark_leakage(
    records: Sequence[Dict[str, Any]],
    benchmark_records: Sequence[Dict[str, Any]],
    threshold: float = 0.78,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records or not benchmark_records:
        return list(records), []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError("scikit-learn is required for benchmark leakage checks.") from exc

    train_texts = [str(r["instruction"]).lower() for r in records]
    benchmark_texts = [str(r["instruction"]).lower() for r in benchmark_records]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    all_matrix = vectorizer.fit_transform(train_texts + benchmark_texts)
    train_matrix = all_matrix[: len(train_texts)]
    benchmark_matrix = all_matrix[len(train_texts) :]
    scores = cosine_similarity(train_matrix, benchmark_matrix)

    kept, removed = [], []
    for record, row in zip(records, scores):
        max_index = int(row.argmax())
        max_score = float(row[max_index])
        if max_score >= threshold:
            removed.append({
                "id": record.get("id"),
                "instruction": record.get("instruction"),
                "benchmark_id": benchmark_records[max_index].get("id"),
                "similarity": round(max_score, 4),
            })
        else:
            kept.append(dict(record))
    return kept, removed


def assign_stratified_splits(
    records: Sequence[Dict[str, Any]],
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> List[Dict[str, Any]]:
    """Assign deterministic category-stratified train/validation/test labels."""
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("category", "unknown"))].append(dict(record))

    assigned: List[Dict[str, Any]] = []
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(round(n * train_fraction)))
        n_val = int(round(n * validation_fraction))
        if n >= 10:
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            if n_train + n_val + n_test > n:
                n_train -= n_train + n_val + n_test - n
        else:
            n_test = max(0, n - n_train - n_val)
        for index, item in enumerate(group):
            if index < n_train:
                item["split"] = "train"
            elif index < n_train + n_val:
                item["split"] = "validation"
            else:
                item["split"] = "test"
            assigned.append(item)
    rng.shuffle(assigned)
    return assigned


def dataset_statistics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    instruction_lengths = [len(str(r.get("instruction", "")).split()) for r in records]
    output_lengths = [len(str(r.get("output", "")).split()) for r in records]
    return {
        "total_examples": len(records),
        "split_counts": dict(Counter(str(r.get("split", "unknown")) for r in records)),
        "category_counts": dict(Counter(str(r.get("category", "unknown")) for r in records)),
        "difficulty_counts": dict(Counter(str(r.get("difficulty", "unknown")) for r in records)),
        "topic_count": len({str(r.get("topic", "")) for r in records}),
        "average_instruction_words": round(sum(instruction_lengths) / len(instruction_lengths), 2) if records else 0,
        "average_output_words": round(sum(output_lengths) / len(output_lengths), 2) if records else 0,
        "min_output_words": min(output_lengths, default=0),
        "max_output_words": max(output_lengths, default=0),
    }


def generate_enhanced_dataset(
    *,
    seed_dataset_path: str | Path,
    topic_plan_path: str | Path,
    benchmark_path: str | Path,
    output_dataset_path: str | Path,
    output_dir: str | Path,
    config: DatasetGenerationConfig | None = None,
    human_review_required: bool = True,
) -> Dict[str, Any]:
    """Generate, validate, de-duplicate, split, and save the enhanced dataset."""
    cfg = config or DatasetGenerationConfig()
    hw = detect_hardware()
    model, tokenizer, teacher_model_id = load_teacher_model(cfg, hw)
    topics = load_topic_plan(topic_plan_path)
    seed_records = load_jsonl(seed_dataset_path)
    benchmark_records = load_jsonl(benchmark_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw_generated: List[Dict[str, Any]] = []
    generation_logs: List[Dict[str, Any]] = []
    next_id = 0
    for topic in topics:
        if len(seed_records) + len(raw_generated) >= cfg.target_examples:
            break
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                topic_records, log = generate_topic_records(model, tokenizer, topic, cfg)
                normalized = normalize_generated_records(topic_records, str(topic["topic"]), next_id)
                if not normalized:
                    raise ValueError("Teacher returned no usable records.")
                next_id += len(normalized)
                raw_generated.extend(normalized)
                log["status"] = "completed"
                log["attempt"] = attempt
                generation_logs.append(log)
                last_error = None
                break
            except Exception as exc:  # keep the long run resumable and auditable
                last_error = exc
                generation_logs.append({
                    "topic": topic.get("topic"),
                    "status": "retrying" if attempt < 3 else "failed",
                    "attempt": attempt,
                    "error": repr(exc),
                })
        if last_error is not None:
            continue

    save_jsonl(raw_generated, output / "raw_teacher_generations.jsonl")
    (output / "teacher_generation_log.json").write_text(json.dumps(generation_logs, indent=2), encoding="utf-8")

    combined = [dict(r) for r in seed_records] + raw_generated
    cleaned, validation_report = validate_and_clean_records(combined, min_output_words=20, max_output_words=260)
    deduplicated, duplicate_removals = remove_near_duplicates(cleaned, cfg.duplicate_similarity_threshold)
    leak_free, leakage_removals = remove_benchmark_leakage(
        deduplicated, benchmark_records, cfg.benchmark_leakage_threshold
    )
    split_records = assign_stratified_splits(leak_free, seed=cfg.seed)

    save_jsonl(split_records, output_dataset_path)
    save_jsonl(split_records, output / "enhanced_dataset_review_copy.jsonl")
    report = {
        "status": "completed_pending_human_review" if human_review_required else "completed",
        "teacher_model_id": teacher_model_id,
        "configuration": asdict(cfg),
        "hardware": hw.to_dict(),
        "seed_examples": len(seed_records),
        "raw_generated_examples": len(raw_generated),
        "validation_report": validation_report.to_dict(),
        "near_duplicates_removed": len(duplicate_removals),
        "benchmark_leakage_removed": len(leakage_removals),
        "final_statistics": dataset_statistics(split_records),
        "human_review_required": human_review_required,
        "human_review_instruction": "Review a stratified sample and all advanced/code records before training.",
    }
    (output / "enhanced_dataset_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "duplicate_removals.json").write_text(json.dumps(duplicate_removals, indent=2), encoding="utf-8")
    (output / "benchmark_leakage_removals.json").write_text(json.dumps(leakage_removals, indent=2), encoding="utf-8")
    return report
