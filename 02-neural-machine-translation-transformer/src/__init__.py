"""Reusable English–Hindi neural machine translation package."""

from .language_detection import LanguageDetection, detect_language
from .translation_pipeline import (
    DirectionResolutionError,
    TranslationPipeline,
    TranslationResult,
    build_default_pipeline,
)

__all__ = [
    "LanguageDetection",
    "detect_language",
    "DirectionResolutionError",
    "TranslationPipeline",
    "TranslationResult",
    "build_default_pipeline",
]
