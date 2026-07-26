from __future__ import annotations

import re
from dataclasses import dataclass

from .text_preprocessing import clean_text

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class LanguageDetection:
    language: str
    latin_characters: int
    devanagari_characters: int
    latin_ratio: float
    devanagari_ratio: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "latin_characters": self.latin_characters,
            "devanagari_characters": self.devanagari_characters,
            "latin_ratio": round(self.latin_ratio, 4),
            "devanagari_ratio": round(self.devanagari_ratio, 4),
            "reason": self.reason,
        }


def detect_language(
    text: str,
    *,
    mixed_script_min_ratio: float = 0.20,
    minimum_letters: int = 2,
) -> LanguageDetection:
    cleaned = clean_text(text)
    latin = len(LATIN_RE.findall(cleaned))
    devanagari = len(DEVANAGARI_RE.findall(cleaned))
    total = latin + devanagari

    if total < minimum_letters:
        return LanguageDetection(
            "uncertain",
            latin,
            devanagari,
            0.0 if total == 0 else latin / total,
            0.0 if total == 0 else devanagari / total,
            "Too little Latin or Devanagari evidence.",
        )

    latin_ratio = latin / total
    devanagari_ratio = devanagari / total

    if latin and devanagari:
        if min(latin_ratio, devanagari_ratio) >= mixed_script_min_ratio:
            return LanguageDetection(
                "mixed",
                latin,
                devanagari,
                latin_ratio,
                devanagari_ratio,
                "Meaningful Latin and Devanagari content is present.",
            )
        dominant = "english" if latin_ratio > devanagari_ratio else "hindi"
        return LanguageDetection(
            dominant,
            latin,
            devanagari,
            latin_ratio,
            devanagari_ratio,
            f"{dominant.title()} script is strongly dominant.",
        )

    language = "english" if latin else "hindi"
    return LanguageDetection(
        language,
        latin,
        devanagari,
        latin_ratio,
        devanagari_ratio,
        f"Only {'Latin' if language == 'english' else 'Devanagari'} letters detected.",
    )
