import pandas as pd

from src.batch_translation import translate_dataframe
from src.confidence_scoring import ConfidenceProxy
from src.translation_model import ModelTranslation
from src.translation_pipeline import TranslationPipeline


class FakeEngine:
    def translate(self, text: str, direction: str) -> ModelTranslation:
        return ModelTranslation(
            translated_text=f"translated:{text}",
            confidence=ConfidenceProxy(0.7, "test", "medium proxy", "test"),
            latency_seconds=0.02,
            model_id="fake/model",
            input_tokens=2,
            output_tokens=2,
            device="cpu",
        )


def test_batch_output_schema():
    dataframe = pd.DataFrame({"text": ["Hello", "नमस्ते"]})
    output, summary = translate_dataframe(
        dataframe,
        text_column="text",
        pipeline=TranslationPipeline(engine=FakeEngine()),
    )
    assert len(output) == 2
    assert summary["successful_rows"] == 2
    assert "confidence_score" in output.columns
