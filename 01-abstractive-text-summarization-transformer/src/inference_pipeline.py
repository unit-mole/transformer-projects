from __future__ import annotations

import threading

from .summarization_model import GenerationSettings, SummaryResult, TransformerSummarizer

_SUMMARIZER: TransformerSummarizer | None = None
_LOCK = threading.Lock()


def get_summarizer() -> TransformerSummarizer:
    global _SUMMARIZER
    if _SUMMARIZER is None:
        with _LOCK:
            if _SUMMARIZER is None:
                _SUMMARIZER = TransformerSummarizer()
    return _SUMMARIZER


def summarize_text(
    text: str,
    settings: GenerationSettings | None = None,
) -> SummaryResult:
    return get_summarizer().summarize(text, settings)
