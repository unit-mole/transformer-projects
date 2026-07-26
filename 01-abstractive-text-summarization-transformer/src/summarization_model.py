from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from .config import configured_model_name
from .text_preprocessing import clean_text, validate_article, word_count


@dataclass(frozen=True)
class GenerationSettings:
    min_length: int = 30
    max_length: int = 120
    num_beams: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True

    def validate(self) -> "GenerationSettings":
        if self.min_length < 1:
            raise ValueError("Minimum summary length must be at least 1 token.")
        if self.max_length <= self.min_length:
            raise ValueError("Maximum summary length must be greater than minimum length.")
        if not 1 <= self.num_beams <= 12:
            raise ValueError("Number of beams must be between 1 and 12.")
        if not 0.1 <= self.length_penalty <= 5.0:
            raise ValueError("Length penalty must be between 0.1 and 5.0.")
        if not 0 <= self.no_repeat_ngram_size <= 6:
            raise ValueError("No-repeat n-gram size must be between 0 and 6.")
        return self


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    inference_seconds: float
    input_words: int
    summary_words: int
    compression_ratio: float
    chunks_processed: int
    model_name: str
    device: str
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TransformerSummarizer:
    """Lazy-loading DistilBART summarizer with token-aware long-text handling."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        max_input_tokens: int = 900,
        chunk_overlap_tokens: int = 64,
        min_article_words: int = 25,
    ) -> None:
        if max_input_tokens < 128:
            raise ValueError("max_input_tokens must be at least 128.")
        if not 0 <= chunk_overlap_tokens < max_input_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than max_input_tokens.")
        self.model_name = model_name or configured_model_name()
        self.max_input_tokens = max_input_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_article_words = min_article_words
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "not-loaded"
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        if self.is_loaded:
            return
        if os.getenv("SKIP_MODEL_LOAD") == "1":
            raise RuntimeError("Model loading is disabled by SKIP_MODEL_LOAD=1.")

        with self._load_lock:
            if self.is_loaded:
                return
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Install torch and transformers from requirements.txt before inference."
                ) from exc

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()

    def _token_chunks(self, text: str) -> list[list[int]]:
        assert self._tokenizer is not None
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= self.max_input_tokens:
            return [token_ids]
        step = self.max_input_tokens - self.chunk_overlap_tokens
        return [
            token_ids[start : start + self.max_input_tokens]
            for start in range(0, len(token_ids), step)
            if token_ids[start : start + self.max_input_tokens]
        ]

    def _generate_from_text(self, text: str, settings: GenerationSettings) -> str:
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "min_length": settings.min_length,
            "max_length": settings.max_length,
            "num_beams": settings.num_beams,
            "length_penalty": settings.length_penalty,
            "no_repeat_ngram_size": settings.no_repeat_ngram_size,
            "early_stopping": settings.early_stopping,
            "do_sample": False,
        }
        with self._torch.inference_mode():
            generated = self._model.generate(**encoded, **generation_kwargs)
        return clean_text(self._tokenizer.decode(generated[0], skip_special_tokens=True))

    def summarize(
        self,
        text: str,
        settings: GenerationSettings | None = None,
    ) -> SummaryResult:
        settings = (settings or GenerationSettings()).validate()
        article = validate_article(text, min_words=self.min_article_words)
        self.load()
        assert self._tokenizer is not None

        started = time.perf_counter()
        token_chunks = self._token_chunks(article)
        chunk_summaries: list[str] = []
        for chunk in token_chunks:
            chunk_text = self._tokenizer.decode(chunk, skip_special_tokens=True)
            chunk_summaries.append(self._generate_from_text(chunk_text, settings))

        if len(chunk_summaries) == 1:
            final_summary = chunk_summaries[0]
        else:
            combined = clean_text(" ".join(chunk_summaries))
            combined_tokens = self._tokenizer.encode(combined, add_special_tokens=False)
            if len(combined_tokens) > self.max_input_tokens:
                combined = self._tokenizer.decode(
                    combined_tokens[: self.max_input_tokens], skip_special_tokens=True
                )
            final_summary = self._generate_from_text(combined, settings)

        elapsed = time.perf_counter() - started
        input_words = word_count(article)
        summary_words = word_count(final_summary)
        return SummaryResult(
            summary=final_summary,
            inference_seconds=elapsed,
            input_words=input_words,
            summary_words=summary_words,
            compression_ratio=summary_words / max(input_words, 1),
            chunks_processed=len(token_chunks),
            model_name=self.model_name,
            device=self._device,
            settings=asdict(settings),
        )
