from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Protocol

from .language_detection import LanguageDetection, detect_language
from .text_preprocessing import clean_text
from .translation_model import MarianTranslationEngine, ModelTranslation


class TranslationEngineProtocol(Protocol):
    def translate(self, text: str, direction: str) -> ModelTranslation: ...


class DirectionResolutionError(ValueError):
    pass


DIRECTION_LABELS = {
    "auto": "Automatic",
    "en_hi": "English → Hindi",
    "hi_en": "Hindi → English",
}
DIRECTION_ALIASES = {
    "automatic": "auto",
    "auto": "auto",
    "english → hindi": "en_hi",
    "english -> hindi": "en_hi",
    "english to hindi": "en_hi",
    "en_hi": "en_hi",
    "en-hi": "en_hi",
    "hindi → english": "hi_en",
    "hindi -> english": "hi_en",
    "hindi to english": "hi_en",
    "hi_en": "hi_en",
    "hi-en": "hi_en",
}


@dataclass(frozen=True)
class TranslationResult:
    original_text: str
    translated_text: str
    detected_language: str
    translation_direction: str
    direction_label: str
    confidence_score: float
    confidence_label: str
    confidence_method: str
    confidence_explanation: str
    latency_seconds: float
    model_id: str
    input_tokens: int
    output_tokens: int
    device: str
    warning: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_direction(direction: str | None) -> str:
    value = (direction or "auto").strip().lower()
    if value not in DIRECTION_ALIASES:
        raise DirectionResolutionError(
            f"Unknown direction '{direction}'. Choose Automatic, English → Hindi, "
            "or Hindi → English."
        )
    return DIRECTION_ALIASES[value]


def resolve_direction(
    text: str,
    requested_direction: str | None = "auto",
) -> tuple[str, LanguageDetection]:
    normalized = normalize_direction(requested_direction)
    detection = detect_language(text)

    if normalized == "en_hi":
        return normalized, detection
    if normalized == "hi_en":
        return normalized, detection

    if detection.language == "english":
        return "en_hi", detection
    if detection.language == "hindi":
        return "hi_en", detection

    raise DirectionResolutionError(
        "Automatic direction is unavailable because the input is mixed or uncertain. "
        "Choose English → Hindi or Hindi → English manually."
    )


class TranslationPipeline:
    def __init__(
        self,
        engine: TranslationEngineProtocol | None = None,
        *,
        max_characters: int = 5000,
    ) -> None:
        self.engine = engine or MarianTranslationEngine()
        self.max_characters = max_characters

    def translate(
        self,
        text: str,
        direction: str | None = "auto",
    ) -> TranslationResult:
        cleaned = clean_text(text, max_characters=self.max_characters)
        if not cleaned:
            raise ValueError("Enter text before requesting a translation.")

        resolved_direction, detection = resolve_direction(cleaned, direction)
        model_result = self.engine.translate(cleaned, resolved_direction)

        return TranslationResult(
            original_text=cleaned,
            translated_text=model_result.translated_text,
            detected_language=detection.language,
            translation_direction=resolved_direction,
            direction_label=DIRECTION_LABELS[resolved_direction],
            confidence_score=round(model_result.confidence.score, 6),
            confidence_label=model_result.confidence.label,
            confidence_method=model_result.confidence.method,
            confidence_explanation=model_result.confidence.explanation,
            latency_seconds=round(model_result.latency_seconds, 6),
            model_id=model_result.model_id,
            input_tokens=model_result.input_tokens,
            output_tokens=model_result.output_tokens,
            device=model_result.device,
            warning=(
                "Confidence is a model-based proxy, not a guarantee. "
                "Human review is required."
            ),
        )


@lru_cache(maxsize=1)
def build_default_pipeline() -> TranslationPipeline:
    return TranslationPipeline()
