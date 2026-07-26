from src.confidence_scoring import ConfidenceProxy
from src.translation_model import ModelTranslation
from src.translation_pipeline import DirectionResolutionError, TranslationPipeline


class FakeEngine:
    def translate(self, text: str, direction: str) -> ModelTranslation:
        output = "नकली अनुवाद" if direction == "en_hi" else "mock translation"
        return ModelTranslation(
            translated_text=output,
            confidence=ConfidenceProxy(
                score=0.8,
                method="test",
                label="higher proxy",
                explanation="test only",
            ),
            latency_seconds=0.01,
            model_id=f"fake/{direction}",
            input_tokens=5,
            output_tokens=4,
            device="cpu",
        )


def test_automatic_english_to_hindi():
    result = TranslationPipeline(engine=FakeEngine()).translate(
        "The report is ready.",
        "Automatic",
    )
    assert result.translation_direction == "en_hi"
    assert result.translated_text == "नकली अनुवाद"


def test_automatic_hindi_to_english():
    result = TranslationPipeline(engine=FakeEngine()).translate(
        "रिपोर्ट तैयार है।",
        "Automatic",
    )
    assert result.translation_direction == "hi_en"
    assert result.translated_text == "mock translation"


def test_mixed_requires_manual_direction():
    pipeline = TranslationPipeline(engine=FakeEngine())
    try:
        pipeline.translate("Hello नमस्ते", "Automatic")
    except DirectionResolutionError:
        pass
    else:
        raise AssertionError("Mixed input should require a manual direction.")
