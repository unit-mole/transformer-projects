import pytest

from src.text_preprocessing import clean_text


def test_clean_text_preserves_devanagari_and_numbers():
    value = "  <b>गुणवत्ता</b>\u00a0रिपोर्ट 2026 तैयार है। "
    assert clean_text(value) == "गुणवत्ता रिपोर्ट 2026 तैयार है।"


def test_clean_text_removes_control_characters():
    assert clean_text("hello\x01 world") == "hello world"


def test_clean_text_rejects_oversized_input():
    with pytest.raises(ValueError):
        clean_text("a" * 11, max_characters=10)
