import os

from src.summarization_model import TransformerSummarizer


def test_model_is_lazy_loaded() -> None:
    summarizer = TransformerSummarizer()
    assert summarizer.is_loaded is False
    assert summarizer.device == "not-loaded"


def test_skip_model_load_guard(monkeypatch) -> None:
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")
    summarizer = TransformerSummarizer()
    try:
        summarizer.load()
    except RuntimeError as exc:
        assert "SKIP_MODEL_LOAD" in str(exc)
    else:
        raise AssertionError("Expected model load to be blocked")
    monkeypatch.delenv("SKIP_MODEL_LOAD", raising=False)
