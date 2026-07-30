"""Build a higher-quality Version 3 instruction dataset for Experiment 2."""
from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from .data_preprocessing import load_jsonl, normalize_text, save_jsonl, validate_and_clean_records


@dataclass
class DatasetV3Report:
    seed_records_loaded: int
    previous_v2_records_loaded: int
    previous_v2_records_reused: int
    curated_topic_records: int
    curated_comparison_records: int
    curated_code_records: int
    curated_workflow_records: int
    rejected_quality: int
    rejected_benchmark_overlap: int
    rejected_near_duplicate: int
    final_records: int
    train_records: int
    validation_records: int
    test_records: int
    output_word_mean: float
    output_word_median: float
    source_counts: Dict[str, int]
    category_counts: Dict[str, int]
    difficulty_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _structured_output(card: Dict[str, str]) -> str:
    return (
        f"Definition: {card['definition']}\n\n"
        f"Intuition: {card['intuition']}\n\n"
        f"Example: {card['example']}\n\n"
        f"Important caveat: {card['caveat']}"
    )


def _topic_records(cards: Sequence[Dict[str, str]]) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for card in cards:
        topic = card["topic"]
        records.extend(
            [
                {
                    "instruction": f"Teach the core idea of {topic} using a definition, intuition, example, and caveat.",
                    "input": "Write for an intermediate ML learner and keep every section technically precise.",
                    "output": _structured_output(card),
                    "category": "Concept explanation",
                    "difficulty": "intermediate",
                    "topic": topic,
                    "source": "expert-curated-v3-topic-card",
                },
                {
                    "instruction": f"Explain {topic} to a beginner without using a circular definition.",
                    "input": "Include a practical example and one limitation in plain language.",
                    "output": (
                        f"{card['intuition']} In formal terms, {card['definition']} "
                        f"For example, {card['example']} A limitation to remember is that {card['caveat'][0].lower() + card['caveat'][1:]}"
                    ),
                    "category": "Beginner-friendly explanation",
                    "difficulty": "beginner",
                    "topic": topic,
                    "source": "expert-curated-v3-topic-card",
                },
                {
                    "instruction": f"Give a concrete, non-confidential quality analytics use of {topic} and explain one risk.",
                    "input": "Use a manufacturing, inspection, complaint, or process setting without private company data.",
                    "output": f"Quality use case: {card['quality_example']}\n\nRisk or limitation: {card['caveat']}",
                    "category": "Quality analytics",
                    "difficulty": "intermediate",
                    "topic": topic,
                    "source": "expert-curated-v3-topic-card",
                },
                {
                    "instruction": f"Answer an interview question about {topic} in a concise but technically complete way.",
                    "input": "State what it is, when it is useful, and one caveat.",
                    "output": (
                        f"{card['definition']} It is useful when {card['example'][0].lower() + card['example'][1:]} "
                        f"The main caveat is that {card['caveat'][0].lower() + card['caveat'][1:]}"
                    ),
                    "category": "Interview-style answer",
                    "difficulty": "intermediate",
                    "topic": topic,
                    "source": "expert-curated-v3-topic-card",
                },
                {
                    "instruction": f"Create a short learning example that demonstrates {topic} and then explain why the example is not sufficient by itself.",
                    "input": "The answer must separate the example from the caveat.",
                    "output": f"Example: {card['example']}\n\nWhy that is not sufficient by itself: {card['caveat']}",
                    "category": "Example generation",
                    "difficulty": "intermediate",
                    "topic": topic,
                    "source": "expert-curated-v3-topic-card",
                },
            ]
        )
    return records


def _comparison_records(comparisons: Sequence[Dict[str, str]]) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for item in comparisons:
        left, right = item["left"], item["right"]
        complete = (
            f"Core difference: {item['core_difference']}\n\n"
            f"Prefer {left}: {item['when_left']}\n\n"
            f"Prefer {right}: {item['when_right']}\n\n"
            f"Caveat: {item['caveat']}"
        )
        records.extend(
            [
                {
                    "instruction": f"Contrast {left} with {right} and give a decision rule for choosing between them.",
                    "input": "Use the headings Core difference, Prefer the first, Prefer the second, and Caveat.",
                    "output": complete,
                    "category": "Algorithm comparison",
                    "difficulty": "intermediate",
                    "topic": f"{left} vs {right}",
                    "source": "expert-curated-v3-comparison",
                },
                {
                    "instruction": f"Give a portfolio-ready interview answer comparing {left} and {right}.",
                    "input": "Do not claim that one is universally better.",
                    "output": (
                        f"{item['core_difference']} {item['when_left']} {item['when_right']} "
                        f"The choice is context-dependent because {item['caveat'][0].lower() + item['caveat'][1:]}"
                    ),
                    "category": "Interview-style answer",
                    "difficulty": "advanced",
                    "topic": f"{left} vs {right}",
                    "source": "expert-curated-v3-comparison",
                },
            ]
        )
    return records


def _code_records(code_examples: Sequence[Dict[str, str]]) -> list[Dict[str, Any]]:
    return [
        {
            "instruction": item["instruction"],
            "input": "Use Python and explain the leakage, evaluation, or reproducibility safeguard after the code.",
            "output": item["output"],
            "category": "Small code example",
            "difficulty": "advanced",
            "topic": item["topic"],
            "source": "expert-curated-v3-code",
        }
        for item in code_examples
    ]


def _workflow_records(workflows: Sequence[Dict[str, str]]) -> list[Dict[str, Any]]:
    categories = ["Data Science workflow", "ML project guidance"]
    return [
        {
            "instruction": item["instruction"],
            "input": "Present actionable steps and include evaluation or governance safeguards.",
            "output": item["output"],
            "category": categories[index % len(categories)],
            "difficulty": "advanced" if index % 3 == 0 else "intermediate",
            "topic": item["topic"],
            "source": "expert-curated-v3-workflow",
        }
        for index, item in enumerate(workflows)
    ]


def _quality_reasons(record: Dict[str, Any], rules: Dict[str, Any]) -> list[str]:
    output = normalize_text(record.get("output", ""))
    source = normalize_text(record.get("source", "")).lower()
    category = normalize_text(record.get("category", ""))
    reasons: list[str] = []
    word_count = len(output.split())
    min_words = 12 if category == "Small code example" else int(rules["min_output_words"])
    if word_count < min_words:
        reasons.append("output_too_short")
    if word_count > int(rules["max_output_words"]):
        reasons.append("output_too_long")
    lowered = output.lower()
    for phrase in rules.get("weak_phrases", []):
        if phrase.lower() in lowered:
            reasons.append(f"weak_phrase:{phrase}")
    for pattern in rules.get("circular_patterns", []):
        try:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                reasons.append("circular_definition")
        except re.error:
            continue
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", output) if s.strip()]
    if len(sentences) >= 3 and len(set(sentences)) / len(sentences) < 0.75:
        reasons.append("repeated_sentences")
    tokens = re.findall(r"[a-zA-Z]+", lowered)
    if len(tokens) >= 40 and len(set(tokens)) / len(tokens) < 0.35:
        reasons.append("low_lexical_diversity")
    if not any(marker in source for marker in rules.get("trusted_source_markers", [])):
        if "example" not in lowered and "caveat" not in lowered and "however" not in lowered:
            reasons.append("untrusted_record_missing_explanation_structure")
    return sorted(set(reasons))


def _max_similarity_to_benchmark(
    instructions: Sequence[str], benchmark_prompts: Sequence[str]
) -> list[float]:
    if not instructions or not benchmark_prompts:
        return [0.0] * len(instructions)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    matrix = vectorizer.fit_transform(list(instructions) + list(benchmark_prompts))
    candidate_matrix = matrix[: len(instructions)]
    benchmark_matrix = matrix[len(instructions) :]
    similarities = cosine_similarity(candidate_matrix, benchmark_matrix)
    return [float(row.max()) if row.size else 0.0 for row in similarities]


def _deduplicate(records: Sequence[Dict[str, Any]], threshold: float) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    exact_seen: set[str] = set()
    exact_unique: list[Dict[str, Any]] = []
    removed: list[Dict[str, Any]] = []
    for record in records:
        key = normalize_text(record.get("instruction", "")).lower()
        if key in exact_seen:
            removed.append({"reason": "exact_duplicate", "instruction": record.get("instruction", "")})
            continue
        exact_seen.add(key)
        exact_unique.append(record)

    if len(exact_unique) < 2:
        return exact_unique, removed
    instructions = [normalize_text(r.get("instruction", "")) for r in exact_unique]
    matrix = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform(instructions)
    similarity = cosine_similarity(matrix)
    keep: list[int] = []
    for index in range(len(exact_unique)):
        if any(similarity[index, previous] >= threshold for previous in keep):
            removed.append(
                {
                    "reason": "near_duplicate",
                    "instruction": exact_unique[index].get("instruction", ""),
                    "max_similarity": max(similarity[index, previous] for previous in keep),
                }
            )
            continue
        keep.append(index)
    return [exact_unique[index] for index in keep], removed


def _assign_splits(records: Sequence[Dict[str, Any]], seed: int = 42) -> list[Dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["category"]), str(record["difficulty"]))].append(dict(record))

    rng = random.Random(seed)
    assigned: list[Dict[str, Any]] = []
    for group_key in sorted(grouped):
        group = grouped[group_key]
        rng.shuffle(group)
        n = len(group)
        if n >= 10:
            n_validation = max(1, round(n * 0.10))
            n_test = max(1, round(n * 0.10))
        elif n >= 5:
            n_validation = 1
            n_test = 1
        else:
            n_validation = 0
            n_test = 0
        for index, record in enumerate(group):
            if index < n_validation:
                record["split"] = "validation"
            elif index < n_validation + n_test:
                record["split"] = "test"
            else:
                record["split"] = "train"
            assigned.append(record)

    rng.shuffle(assigned)
    for index, record in enumerate(assigned):
        record["id"] = f"ml_ds_v3_{index:04d}"
    return assigned


def _write_review_sample(records: Sequence[Dict[str, Any]], path: Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    by_category: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[str(record["category"])].append(record)
    sample: list[Dict[str, Any]] = []
    for category in sorted(by_category):
        candidates = by_category[category][:]
        rng.shuffle(candidates)
        sample.extend(candidates[: min(4, len(candidates))])
    advanced = [r for r in records if r.get("difficulty") == "advanced"]
    rng.shuffle(advanced)
    known_ids = {r["id"] for r in sample}
    sample.extend([r for r in advanced if r["id"] not in known_ids][:20])
    fields = ["id", "instruction", "input", "output", "category", "difficulty", "topic", "source", "split"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in sample])


def build_dataset_v3(
    *,
    seed_dataset_path: str | Path,
    previous_v2_path: str | Path,
    benchmark_path: str | Path,
    topic_cards_path: str | Path,
    comparisons_path: str | Path,
    code_examples_path: str | Path,
    workflows_path: str | Path,
    rules_path: str | Path,
    output_dataset_path: str | Path,
    output_report_dir: str | Path,
    reuse_teacher_records: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build v3 from trusted seed records and expert-curated anchors.

    The default intentionally excludes Experiment 1 teacher-generated records.
    They can be re-enabled, but only records that pass strict quality rules are retained.
    """
    rules = _load_json(rules_path)
    seed_records = load_jsonl(seed_dataset_path)
    previous_records = load_jsonl(previous_v2_path) if Path(previous_v2_path).exists() else []
    benchmark = load_jsonl(benchmark_path)
    benchmark_prompts = [normalize_text(row.get("instruction") or row.get("prompt")) for row in benchmark]

    candidates: list[Dict[str, Any]] = []
    candidates.extend(seed_records)
    reused_previous = 0
    rejection_log: list[Dict[str, Any]] = []

    if reuse_teacher_records:
        for record in previous_records:
            reasons = _quality_reasons(record, rules)
            if reasons:
                rejection_log.append(
                    {
                        "stage": "previous_v2_quality_filter",
                        "id": record.get("id", ""),
                        "instruction": record.get("instruction", ""),
                        "reasons": reasons,
                    }
                )
                continue
            candidates.append(record)
            reused_previous += 1

    topic_records = _topic_records(_load_json(topic_cards_path))
    comparison_records = _comparison_records(_load_json(comparisons_path))
    code_records = _code_records(_load_json(code_examples_path))
    workflow_records = _workflow_records(_load_json(workflows_path))
    candidates.extend(topic_records + comparison_records + code_records + workflow_records)

    quality_pass: list[Dict[str, Any]] = []
    for record in candidates:
        reasons = _quality_reasons(record, rules)
        source = normalize_text(record.get("source", "")).lower()
        trusted = any(marker in source for marker in rules.get("trusted_source_markers", []))
        if reasons and not trusted:
            rejection_log.append(
                {
                    "stage": "all_candidate_quality_filter",
                    "id": record.get("id", ""),
                    "instruction": record.get("instruction", ""),
                    "reasons": reasons,
                }
            )
            continue
        quality_pass.append(record)

    similarities = _max_similarity_to_benchmark(
        [normalize_text(r.get("instruction", "")) for r in quality_pass], benchmark_prompts
    )
    leakage_pass: list[Dict[str, Any]] = []
    leakage_removed = 0
    for record, similarity in zip(quality_pass, similarities):
        if similarity >= float(rules["benchmark_similarity_threshold"]):
            leakage_removed += 1
            rejection_log.append(
                {
                    "stage": "benchmark_leakage_filter",
                    "id": record.get("id", ""),
                    "instruction": record.get("instruction", ""),
                    "max_similarity": round(similarity, 6),
                }
            )
            continue
        record = dict(record)
        record["benchmark_max_similarity"] = round(similarity, 6)
        leakage_pass.append(record)

    deduplicated, duplicate_removed = _deduplicate(
        leakage_pass, float(rules["near_duplicate_threshold"])
    )
    rejection_log.extend({"stage": "deduplication", **row} for row in duplicate_removed)

    cleaned, validation_report = validate_and_clean_records(
        deduplicated,
        min_output_words=12,
        max_output_words=int(rules["max_output_words"]),
    )
    assigned = _assign_splits(cleaned, seed=seed)
    output_dataset = Path(output_dataset_path)
    save_jsonl(assigned, output_dataset)

    report_dir = Path(output_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_v3_rejection_log.json").write_text(
        json.dumps(rejection_log, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    _write_review_sample(assigned, report_dir / "dataset_v3_review_sample.csv", seed=seed)

    words = sorted(len(str(row["output"]).split()) for row in assigned)
    median = words[len(words) // 2] if words else 0
    split_counts = Counter(str(row["split"]) for row in assigned)
    report = DatasetV3Report(
        seed_records_loaded=len(seed_records),
        previous_v2_records_loaded=len(previous_records),
        previous_v2_records_reused=reused_previous,
        curated_topic_records=len(topic_records),
        curated_comparison_records=len(comparison_records),
        curated_code_records=len(code_records),
        curated_workflow_records=len(workflow_records),
        rejected_quality=sum(row.get("stage", "").endswith("quality_filter") for row in rejection_log),
        rejected_benchmark_overlap=leakage_removed,
        rejected_near_duplicate=len(duplicate_removed),
        final_records=len(assigned),
        train_records=split_counts.get("train", 0),
        validation_records=split_counts.get("validation", 0),
        test_records=split_counts.get("test", 0),
        output_word_mean=round(sum(words) / len(words), 2) if words else 0.0,
        output_word_median=float(median),
        source_counts=dict(Counter(str(row["source"]) for row in assigned)),
        category_counts=dict(Counter(str(row["category"]) for row in assigned)),
        difficulty_counts=dict(Counter(str(row["difficulty"]) for row in assigned)),
    )
    payload = {
        "status": "completed",
        "dataset_path": str(output_dataset.resolve()),
        "reuse_teacher_records": reuse_teacher_records,
        "report": report.to_dict(),
        "base_validation_report": validation_report.to_dict(),
        "review_sample": str((report_dir / "dataset_v3_review_sample.csv").resolve()),
        "rejection_log": str((report_dir / "dataset_v3_rejection_log.json").resolve()),
        "important_note": "benchmark_prompts_v2.jsonl was used only for leakage screening, never as training data.",
    }
    (report_dir / "dataset_v3_quality_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload
