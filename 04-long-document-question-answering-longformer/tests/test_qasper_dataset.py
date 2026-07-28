from __future__ import annotations

import json
from pathlib import Path

from src.qasper_dataset import build_paper_text, find_answer_span, flatten_qasper_split


def test_build_paper_text_and_find_span() -> None:
    record = {
        "title": "Quality Report",
        "abstract": "This report describes a supplier issue.",
        "full_text": {
            "section_name": ["Root Cause"],
            "paragraphs": [["The confirmed root cause was curing-time variation."]],
        },
    }
    text, metadata = build_paper_text(record)
    assert "Root Cause" in text
    assert find_answer_span(text, "curing-time variation") is not None
    assert any(item["kind"] == "paragraph" for item in metadata)


def test_flatten_qasper_keeps_contiguous_extractive_answers(tmp_path: Path) -> None:
    raw = {
        "paper-1": {
            "title": "A Paper",
            "abstract": "Short abstract.",
            "full_text": {
                "section_name": ["Results"],
                "paragraphs": [["The measured improvement was 14 percent."]],
            },
            "qas": {
                "question": ["What was the measured improvement?"],
                "question_id": ["q1"],
                "answers": [
                    [
                        {
                            "answer": {
                                "unanswerable": False,
                                "yes_no": False,
                                "extractive_spans": ["14 percent"],
                                "free_form_answer": "",
                                "evidence": ["The measured improvement was 14 percent."],
                                "highlighted_evidence": [],
                            }
                        }
                    ]
                ],
            },
        }
    }
    path = tmp_path / "qasper.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    frame = flatten_qasper_split(path, "validation")
    assert len(frame) == 1
    assert frame.loc[0, "primary_answer"] == "14 percent"
    assert frame.loc[0, "answer_start"] >= 0


def test_flatten_qasper_supports_raw_list_qas(tmp_path: Path) -> None:
    raw = {
        "paper-raw": {
            "title": "Raw Paper",
            "abstract": "Abstract.",
            "full_text": [
                {
                    "section_name": "Findings",
                    "paragraphs": ["The selected threshold was 0.75."],
                }
            ],
            "qas": [
                {
                    "question": "What threshold was selected?",
                    "question_id": "raw-q1",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "yes_no": None,
                                "extractive_spans": ["0.75"],
                                "free_form_answer": "",
                                "evidence": ["The selected threshold was 0.75."],
                                "highlighted_evidence": [],
                            }
                        }
                    ],
                }
            ],
        }
    }
    path = tmp_path / "qasper_raw.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    frame = flatten_qasper_split(path, "train")
    assert len(frame) == 1
    assert frame.loc[0, "primary_answer"] == "0.75"


class _WhitespaceTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, object]:
        offsets = []
        cursor = 0
        for token in text.split():
            start = text.find(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def test_controlled_context_variants_cover_targets() -> None:
    from src.qasper_dataset import build_controlled_context_variants
    import pandas as pd

    words = [f"token{i}" for i in range(120)]
    document = " ".join(words)
    answer = "token70"
    start = document.index(answer)
    frame = pd.DataFrame(
        [
            {
                "example_id": "controlled",
                "document": document,
                "question": "Which token?",
                "answer_start": start,
                "answer_end": start + len(answer),
                "document_character_count": len(document),
                "answer_character_ratio": start / len(document),
            }
        ]
    )
    variants = build_controlled_context_variants(
        frame,
        _WhitespaceTokenizer(),
        target_token_lengths=(20, 40, 80),
        maximum_base_examples=1,
    )
    assert set(variants["controlled_target_tokens"]) == {20, 40, 80}
    for _, row in variants.iterrows():
        assert row["document"][row["answer_start"] : row["answer_end"]] == answer
