from __future__ import annotations

from dataclasses import dataclass

from .baselines import lead3_summary, textrank_summary
from .summarization_model import GenerationSettings, SummaryResult, TransformerSummarizer
from .text_preprocessing import validate_article, word_count


@dataclass(frozen=True)
class BaselineResult:
    method: str
    summary: str
    input_words: int
    summary_words: int
    compression_ratio: float


def run_baseline(text: str, method: str = "lead3", max_sentences: int = 3) -> BaselineResult:
    article = validate_article(text)
    if method == "lead3":
        summary = lead3_summary(article, max_sentences=max_sentences)
        label = "Lead-3"
    elif method == "textrank":
        summary = textrank_summary(article, max_sentences=max_sentences)
        label = "TextRank-style"
    else:
        raise ValueError("method must be 'lead3' or 'textrank'.")
    input_words = word_count(article)
    summary_words = word_count(summary)
    return BaselineResult(
        method=label,
        summary=summary,
        input_words=input_words,
        summary_words=summary_words,
        compression_ratio=summary_words / max(input_words, 1),
    )


def run_transformer(
    text: str,
    settings: GenerationSettings | None = None,
    summarizer: TransformerSummarizer | None = None,
) -> SummaryResult:
    engine = summarizer or TransformerSummarizer()
    return engine.summarize(text, settings)
