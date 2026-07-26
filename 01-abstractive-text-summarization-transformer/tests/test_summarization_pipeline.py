from src.baselines import lead3_summary, textrank_summary
from src.summarization_model import GenerationSettings

TEXT = (
    "The team found delayed investigations. "
    "Analysts standardized the data. "
    "A dashboard highlighted emerging issues. "
    "Review time declined after implementation."
)


def test_lead3_baseline() -> None:
    summary = lead3_summary(TEXT)
    assert summary.startswith("The team found delayed investigations.")
    assert "Review time declined" not in summary


def test_textrank_returns_nonempty_summary() -> None:
    assert textrank_summary(TEXT, max_sentences=2)


def test_generation_settings_validation() -> None:
    assert GenerationSettings(num_beams=4).validate().num_beams == 4
    try:
        GenerationSettings(min_length=50, max_length=40).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid length settings to fail")
