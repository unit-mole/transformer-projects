from __future__ import annotations

from src.config import InferenceConfig
from src.inference_pipeline import LongDocumentQAPipeline
from src.schemas import SpanCandidate


class FakeModel:
    model_max_length = 4096

    def predict(self, question, context, max_length=None, stride=None, max_answer_tokens=None):
        answer = "Priya Raman"
        start = context.index(answer)
        candidate = SpanCandidate(
            answer=answer,
            start_char=start,
            end_char=start + len(answer),
            raw_score=15.0,
            confidence_proxy=0.25,
            feature_index=0,
            start_token=8,
            end_token=9,
        )
        return {
            "candidate": candidate,
            "candidates": [candidate],
            "window_count": 2,
            "runtime_max_length": max_length or 1024,
            "runtime_stride": stride or 128,
            "model_max_length": 4096,
            "device": "cpu",
        }

    def count_context_tokens(self, context):
        return len(context.split())


def test_pipeline_returns_grounded_result() -> None:
    document = (
        "Investigation summary.\n\n"
        "Priya Raman was assigned as the CAPA owner. The action was verified."
    )
    pipeline = LongDocumentQAPipeline(
        config=InferenceConfig(max_length=1024, stride=128),
        model=FakeModel(),
    )
    result = pipeline.answer(
        "Who was assigned as the CAPA owner?",
        document,
        source_name="sample.txt",
        max_length=1024,
        stride=128,
    )

    assert result.answer == "Priya Raman"
    assert result.paragraph_index == 1
    assert "<mark>Priya Raman</mark>" in result.highlighted_evidence_html
    assert result.window_count == 2
    assert result.confidence_proxy == 0.25
