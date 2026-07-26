from __future__ import annotations

import time
from typing import Any, Optional

from .confidence_scoring import confidence_label
from .config import InferenceConfig
from .document_chunking import locate_supporting_paragraph
from .evidence_highlighting import highlight_answer_in_paragraph
from .qa_model import LongformerQAModel
from .schemas import QAResult, SpanCandidate
from .text_preprocessing import count_words, normalize_text


class InferenceValidationError(ValueError):
    """Raised for invalid document or question input."""


class LongDocumentQAPipeline:
    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.config = (config or InferenceConfig()).validate()
        self.model = model or LongformerQAModel(self.config)

    def _validate(self, question: str, document_text: str) -> tuple[str, str]:
        clean_question = normalize_text(question, preserve_paragraphs=False)
        clean_document = normalize_text(document_text)
        if not clean_question:
            raise InferenceValidationError("Enter a question about the document.")
        if len(clean_question) > 1_000:
            raise InferenceValidationError(
                "The question is too long. Use a focused question under 1,000 characters."
            )
        if not clean_document:
            raise InferenceValidationError("The document contains no readable text.")
        if len(clean_document) > self.config.max_document_characters:
            raise InferenceValidationError(
                f"The document exceeds {self.config.max_document_characters:,} characters."
            )
        return clean_question, clean_document

    def answer(
        self,
        question: str,
        document_text: str,
        source_name: str = "document",
        max_length: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> QAResult:
        question, document_text = self._validate(question, document_text)
        started = time.perf_counter()

        prediction = self.model.predict(
            question=question,
            context=document_text,
            max_length=max_length,
            stride=stride,
            max_answer_tokens=self.config.max_answer_tokens,
        )
        latency = time.perf_counter() - started
        candidate: Optional[SpanCandidate] = prediction.get("candidate")
        warnings: list[str] = []

        try:
            document_token_count = int(self.model.count_context_tokens(document_text))
        except Exception:
            document_token_count = None
            warnings.append("Exact document token count could not be calculated.")

        if prediction.get("window_count", 1) > 1:
            warnings.append(
                "The document exceeded one runtime window and was processed with "
                "overlapping token windows."
            )

        if candidate is None:
            warnings.append(
                "The model did not produce a valid context span. Review the question "
                "or use a document containing explicit supporting text."
            )
            return QAResult(
                answer="No supported answer span was found.",
                confidence_proxy=0.0,
                confidence_label="no valid span",
                supporting_paragraph="",
                highlighted_evidence_html=(
                    "<div class='evidence-box'><strong>No evidence highlighted.</strong> "
                    "The model did not return a valid span.</div>"
                ),
                paragraph_index=None,
                answer_start_char=None,
                answer_end_char=None,
                model_id=self.config.model_id,
                model_max_length=int(prediction.get("model_max_length", 4096)),
                requested_max_length=int(
                    prediction.get("runtime_max_length", max_length or self.config.max_length)
                ),
                window_count=int(prediction.get("window_count", 0)),
                document_character_count=len(document_text),
                document_word_count=count_words(document_text),
                document_token_count=document_token_count,
                latency_seconds=latency,
                source_name=source_name,
                warnings=warnings,
                diagnostics={
                    "device": prediction.get("device"),
                    "stride": prediction.get("runtime_stride"),
                    "candidate_count": len(prediction.get("candidates", [])),
                },
            )

        paragraph, paragraphs = locate_supporting_paragraph(
            document_text,
            candidate.start_char,
            candidate.end_char,
        )
        if paragraph is None:
            warnings.append(
                "The predicted answer could not be mapped to a paragraph boundary."
            )

        if candidate.confidence_proxy < self.config.minimum_confidence_proxy:
            warnings.append(
                "The model confidence proxy is very low. Treat the answer as uncertain."
            )

        highlighted = highlight_answer_in_paragraph(
            paragraph=paragraph,
            answer=candidate.answer,
            answer_start_in_document=candidate.start_char,
            answer_end_in_document=candidate.end_char,
        )

        return QAResult(
            answer=candidate.answer,
            confidence_proxy=float(candidate.confidence_proxy),
            confidence_label=confidence_label(candidate.confidence_proxy),
            supporting_paragraph=paragraph.text if paragraph else "",
            highlighted_evidence_html=highlighted,
            paragraph_index=paragraph.chunk_id if paragraph else None,
            answer_start_char=candidate.start_char,
            answer_end_char=candidate.end_char,
            model_id=self.config.model_id,
            model_max_length=int(prediction.get("model_max_length", 4096)),
            requested_max_length=int(
                prediction.get("runtime_max_length", max_length or self.config.max_length)
            ),
            window_count=int(prediction.get("window_count", 1)),
            document_character_count=len(document_text),
            document_word_count=count_words(document_text),
            document_token_count=document_token_count,
            latency_seconds=latency,
            source_name=source_name,
            warnings=warnings,
            diagnostics={
                "device": prediction.get("device"),
                "stride": prediction.get("runtime_stride"),
                "raw_span_score": candidate.raw_score,
                "feature_index": candidate.feature_index,
                "candidate_count": len(prediction.get("candidates", [])),
                "paragraph_count": len(paragraphs),
                "confidence_note": (
                    "This is an uncalibrated proxy derived from start/end token "
                    "probabilities, not a guarantee of correctness."
                ),
            },
        )
