from __future__ import annotations

import json
import random
import re
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

QASPER_TRAIN_DEV_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
    "qasper-train-dev-v0.3.tgz"
)
QASPER_TEST_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
    "qasper-test-and-evaluator-v0.3.tgz"
)
QASPER_FILES = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}

_WS = re.compile(r"\s+")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def _download(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)
    return destination


def download_qasper(raw_directory: str | Path, force: bool = False) -> dict[str, Path]:
    """Download and safely extract the official QASPER v0.3 archives."""
    raw_directory = Path(raw_directory)
    archive_directory = raw_directory / "archives"
    extracted_directory = raw_directory / "extracted"
    extracted_directory.mkdir(parents=True, exist_ok=True)

    train_archive = _download(
        QASPER_TRAIN_DEV_URL,
        archive_directory / "qasper-train-dev-v0.3.tgz",
        force=force,
    )
    test_archive = _download(
        QASPER_TEST_URL,
        archive_directory / "qasper-test-and-evaluator-v0.3.tgz",
        force=force,
    )

    if force or not (extracted_directory / QASPER_FILES["train"]).exists():
        _safe_extract(train_archive, extracted_directory)
    if force or not (extracted_directory / QASPER_FILES["test"]).exists():
        _safe_extract(test_archive, extracted_directory)

    paths: dict[str, Path] = {}
    for split, filename in QASPER_FILES.items():
        matches = list(extracted_directory.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Could not find {filename} after extraction.")
        paths[split] = matches[0]
    return paths


def build_paper_text(record: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Create a readable paper text and preserve paragraph offsets."""
    blocks: list[str] = []
    metadata: list[dict[str, Any]] = []

    def append_block(text: Any, kind: str, section: str = "") -> None:
        value = str(text or "").strip()
        if not value:
            return
        if blocks:
            blocks.append("\n\n")
        start = sum(len(item) for item in blocks)
        blocks.append(value)
        end = start + len(value)
        metadata.append(
            {
                "paragraph_index": len(metadata),
                "kind": kind,
                "section": section,
                "text": value,
                "start_char": start,
                "end_char": end,
            }
        )

    append_block(record.get("title", ""), "title", "Title")
    append_block(record.get("abstract", ""), "abstract", "Abstract")

    full_text = record.get("full_text") or {}
    if isinstance(full_text, list):
        sections = full_text
    else:
        names = full_text.get("section_name", []) or []
        paragraphs = full_text.get("paragraphs", []) or []
        sections = [
            {"section_name": name, "paragraphs": section_paragraphs}
            for name, section_paragraphs in zip(names, paragraphs)
        ]

    for section in sections:
        section_name = str(section.get("section_name", "")).strip()
        if section_name:
            append_block(section_name, "section_heading", section_name)
        for paragraph in section.get("paragraphs", []) or []:
            append_block(paragraph, "paragraph", section_name)

    return "".join(blocks), metadata


def _whitespace_pattern(text: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in _WS.split(text.strip()) if part]
    return re.compile(r"\s+".join(parts), flags=re.IGNORECASE)


def find_answer_span(document: str, answer: str) -> tuple[int, int] | None:
    answer = str(answer or "").strip()
    if not answer:
        return None
    start = document.find(answer)
    if start >= 0:
        return start, start + len(answer)
    start = document.lower().find(answer.lower())
    if start >= 0:
        return start, start + len(answer)
    match = _whitespace_pattern(answer).search(document)
    if match:
        return match.start(), match.end()
    return None


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        key = _WS.sub(" ", clean).lower()
        if clean and key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def _iter_qas(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    qas = record.get("qas") or {}
    if isinstance(qas, list):
        yield from qas
        return

    questions = qas.get("question", []) or []
    question_ids = qas.get("question_id", []) or []
    answers = qas.get("answers", []) or []
    for index, question in enumerate(questions):
        yield {
            "question": question,
            "question_id": question_ids[index] if index < len(question_ids) else str(index),
            "answers": answers[index] if index < len(answers) else [],
        }


def _answer_payload(annotation: Any) -> dict[str, Any]:
    if not isinstance(annotation, dict):
        return {}
    payload = annotation.get("answer", annotation)
    return payload if isinstance(payload, dict) else {}


def flatten_qasper_split(
    raw_json_path: str | Path,
    split: str,
    maximum_examples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Flatten QASPER into answerable, contiguous extractive QA examples.

    QASPER includes free-form, yes/no, unanswerable and multi-span answers. This
    project deliberately keeps only questions with at least one contiguous
    extractive answer found in the reconstructed paper text, because the model
    is an extractive span predictor.
    """
    raw_json_path = Path(raw_json_path)
    papers = json.loads(raw_json_path.read_text(encoding="utf-8"))
    if isinstance(papers, list):
        iterable = [(str(index), paper) for index, paper in enumerate(papers)]
    else:
        iterable = list(papers.items())

    rows: list[dict[str, Any]] = []
    for paper_id, record in iterable:
        document, paragraphs = build_paper_text(record)
        if not document:
            continue
        for qa in _iter_qas(record):
            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            extractive_answers: list[str] = []
            evidence: list[str] = []
            for annotation in qa.get("answers", []) or []:
                payload = _answer_payload(annotation)
                if payload.get("unanswerable") is True:
                    continue
                if payload.get("yes_no") is True:
                    continue
                spans = payload.get("extractive_spans", []) or []
                # The model predicts one contiguous span. Multi-span annotations are
                # excluded rather than incorrectly treating each fragment as a complete answer.
                if len(spans) != 1:
                    continue
                extractive_answers.append(spans[0])
                evidence.extend(
                    item
                    for item in (payload.get("evidence", []) or [])
                    if item and not str(item).startswith("FLOAT SELECTED")
                )
                evidence.extend(
                    item
                    for item in (payload.get("highlighted_evidence", []) or [])
                    if item and not str(item).startswith("FLOAT SELECTED")
                )

            references = _deduplicate(extractive_answers)
            resolved: list[tuple[str, int, int]] = []
            for answer in references:
                span = find_answer_span(document, answer)
                if span:
                    resolved.append((document[span[0] : span[1]], span[0], span[1]))
            if not resolved:
                continue

            primary_answer, answer_start, answer_end = resolved[0]
            reference_answers = _deduplicate(item[0] for item in resolved)
            reference_evidence = _deduplicate(evidence)
            rows.append(
                {
                    "example_id": f"{split}-{paper_id}-{qa.get('question_id', len(rows))}",
                    "split": split,
                    "paper_id": str(paper_id),
                    "title": str(record.get("title", "")).strip(),
                    "question": question,
                    "document": document,
                    "primary_answer": primary_answer,
                    "answer_start": int(answer_start),
                    "answer_end": int(answer_end),
                    "reference_answers_json": json.dumps(reference_answers, ensure_ascii=False),
                    "reference_evidence_json": json.dumps(reference_evidence, ensure_ascii=False),
                    "paragraph_metadata_json": json.dumps(paragraphs, ensure_ascii=False),
                    "document_character_count": len(document),
                    "answer_character_ratio": answer_start / max(len(document), 1),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if maximum_examples is not None:
        frame = frame.head(maximum_examples).copy()
    return frame


def save_prepared_split(frame: pd.DataFrame, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        frame.to_parquet(destination, index=False)
    elif destination.suffix in {".jsonl", ".json"}:
        frame.to_json(destination, orient="records", lines=True, force_ascii=False)
    else:
        frame.to_csv(destination, index=False)


def prepare_qasper_dataset(
    project_root: str | Path,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    seed: int = 42,
    force_download: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root)
    raw_directory = project_root / "data" / "raw" / "qasper"
    processed_directory = project_root / "data" / "processed" / "qasper"
    output_directory = project_root / "outputs"
    processed_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = download_qasper(raw_directory, force=force_download)
    train = flatten_qasper_split(paths["train"], "train", train_limit, seed)
    validation = flatten_qasper_split(
        paths["validation"], "validation", validation_limit, seed
    )

    save_prepared_split(train, processed_directory / "qasper_train_extractive.parquet")
    save_prepared_split(
        validation,
        processed_directory / "qasper_validation_extractive.parquet",
    )
    save_prepared_split(train, processed_directory / "qasper_train_extractive.jsonl")
    save_prepared_split(
        validation,
        processed_directory / "qasper_validation_extractive.jsonl",
    )

    summary = {
        "status": "completed",
        "dataset": "QASPER v0.3",
        "task_subset": "contiguous extractive answers only",
        "train_examples": int(len(train)),
        "validation_examples": int(len(validation)),
        "train_papers": int(train["paper_id"].nunique()) if not train.empty else 0,
        "validation_papers": int(validation["paper_id"].nunique())
        if not validation.empty
        else 0,
        "train_average_characters": float(train["document_character_count"].mean())
        if not train.empty
        else None,
        "validation_average_characters": float(
            validation["document_character_count"].mean()
        )
        if not validation.empty
        else None,
        "seed": seed,
        "source_urls": [QASPER_TRAIN_DEV_URL, QASPER_TEST_URL],
        "license": "CC BY 4.0",
    }
    (output_directory / "qasper_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def load_prepared_split(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".json"}:
        return pd.read_json(path, orient="records", lines=True)
    return pd.read_csv(path)


def select_evaluation_sample(
    frame: pd.DataFrame,
    maximum_examples: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a reproducible sample balanced across answer-position bands."""
    if frame.empty:
        return frame.copy()
    working = frame.copy().reset_index(drop=True)
    working["_source_index"] = working.index
    bins = [-0.001, 0.125, 0.25, 0.50, 0.75, 1.0]
    labels = ["0-12.5%", "12.5-25%", "25-50%", "50-75%", "75-100%"]
    working["answer_position_band"] = pd.cut(
        working["answer_character_ratio"], bins=bins, labels=labels, include_lowest=True
    ).astype(str)
    per_group = max(1, maximum_examples // len(labels))
    sampled: list[pd.DataFrame] = []
    for _, group in working.groupby("answer_position_band", observed=False):
        if group.empty:
            continue
        sampled.append(group.sample(n=min(per_group, len(group)), random_state=seed))
    result = pd.concat(sampled, ignore_index=True) if sampled else working.head(0)
    if len(result) < maximum_examples:
        chosen = set(result["_source_index"].tolist()) if "_source_index" in result else set()
        remaining = working.loc[~working["_source_index"].isin(chosen)]
        if not remaining.empty:
            extra = remaining.sample(
                n=min(maximum_examples - len(result), len(remaining)),
                random_state=seed + 1,
            )
            result = pd.concat([result, extra], ignore_index=True)
    result = result.head(maximum_examples).sample(frac=1, random_state=seed).reset_index(drop=True)
    return result.drop(columns=["_source_index"], errors="ignore")


def build_controlled_context_variants(
    frame: pd.DataFrame,
    tokenizer: Any,
    target_token_lengths: tuple[int, ...] = (384, 768, 1536, 3072, 4608),
    maximum_base_examples: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Create answer-preserving contexts for controlled length analysis.

    Each selected source example is converted into multiple token-length variants
    centered around the reference answer. Candidates long enough for every target
    length are preferred so all context buckets are represented. Natural
    full-document metrics remain reported separately.
    """
    if frame.empty:
        return frame.copy()
    candidates = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    selected_examples = 0
    maximum_target = max(target_token_lengths)
    deferred: list[tuple[dict[str, Any], Any, int, int, int]] = []

    def add_variants(
        row: dict[str, Any],
        offsets: Any,
        total_tokens: int,
        answer_token_start: int,
        answer_token_end: int,
    ) -> bool:
        created = False
        document = str(row["document"])
        answer_start = int(row["answer_start"])
        answer_end = int(row["answer_end"])
        for target in target_token_lengths:
            if total_tokens < target:
                continue
            desired_start = answer_token_start - target // 2
            token_start = max(0, min(desired_start, total_tokens - target))
            token_end = min(total_tokens, token_start + target)
            if not (token_start <= answer_token_start <= answer_token_end < token_end):
                continue
            char_start = int(offsets[token_start][0])
            char_end = int(offsets[token_end - 1][1])
            context = document[char_start:char_end]
            adjusted_start = answer_start - char_start
            adjusted_end = answer_end - char_start
            if adjusted_start < 0 or adjusted_end > len(context):
                continue
            variant = dict(row)
            variant["source_example_id"] = row["example_id"]
            variant["example_id"] = f"{row['example_id']}-controlled-{target}"
            variant["document"] = context
            variant["answer_start"] = int(adjusted_start)
            variant["answer_end"] = int(adjusted_end)
            variant["document_character_count"] = len(context)
            variant["answer_character_ratio"] = adjusted_start / max(len(context), 1)
            variant["controlled_target_tokens"] = int(target)
            variant["controlled_total_source_tokens"] = int(total_tokens)
            rows.append(variant)
            created = True
        return created

    for row in candidates.to_dict(orient="records"):
        document = str(row["document"])
        encoded = tokenizer(
            document,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        offsets = encoded["offset_mapping"]
        total_tokens = len(offsets)
        if total_tokens == 0:
            continue
        answer_start = int(row["answer_start"])
        answer_end = int(row["answer_end"])
        answer_token_start = None
        answer_token_end = None
        for index, (start_char, end_char) in enumerate(offsets):
            if answer_token_start is None and int(start_char) <= answer_start < int(end_char):
                answer_token_start = index
            if int(start_char) < answer_end <= int(end_char):
                answer_token_end = index
                break
        if answer_token_start is None:
            continue
        if answer_token_end is None:
            answer_token_end = answer_token_start
        payload = (row, offsets, total_tokens, answer_token_start, answer_token_end)
        if total_tokens >= maximum_target and selected_examples < maximum_base_examples:
            if add_variants(*payload):
                selected_examples += 1
        else:
            deferred.append(payload)
        if selected_examples >= maximum_base_examples:
            break

    if selected_examples < maximum_base_examples:
        for payload in deferred:
            if add_variants(*payload):
                selected_examples += 1
            if selected_examples >= maximum_base_examples:
                break
    return pd.DataFrame(rows)
