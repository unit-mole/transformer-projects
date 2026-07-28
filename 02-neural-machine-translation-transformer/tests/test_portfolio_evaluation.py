from __future__ import annotations

import pandas as pd
import pytest

from src.portfolio_evaluation import (
    _normalized_numbers,
    _script_ratios,
    compute_corpus_metrics,
    summarize_manual_review,
)


def test_normalized_numbers_support_devanagari_digits() -> None:
    assert _normalized_numbers("Order 123 costs 45.5") == ["123", "45.5"]
    assert _normalized_numbers("ऑर्डर १२३ की कीमत ४५.५ है") == ["123", "45.5"]


def test_script_ratios() -> None:
    hindi = _script_ratios("यह एक परीक्षण है")
    english = _script_ratios("This is a test")
    assert hindi["devanagari"] > 0.9
    assert english["latin"] > 0.9


def test_corpus_metrics_identical_text_is_high() -> None:
    pytest.importorskip("sacrebleu")
    metrics = compute_corpus_metrics(
        ["This is a test.", "Another sentence."],
        ["This is a test.", "Another sentence."],
    )
    assert metrics["sacrebleu"] > 99
    assert metrics["chrf"] > 99
    assert metrics["ter"] == 0
    assert "tok:13a" in metrics["sacrebleu_signature"]


def test_manual_review_awaiting_status() -> None:
    frame = pd.DataFrame(
        {
            "human_error_category": ["", None],
            "human_severity": ["", ""],
            "human_translation_quality": ["", ""],
            "human_notes": ["", ""],
            "sentence_chrf": [10.0, 20.0],
        }
    )
    result = summarize_manual_review(frame)
    assert result["status"] == "awaiting_human_review"
