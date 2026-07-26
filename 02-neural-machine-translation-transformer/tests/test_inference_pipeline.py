def test_public_import_surface():
    from src.inference_pipeline import (
        TranslationPipeline,
        resolve_direction,
        translate_dataframe,
    )

    assert TranslationPipeline is not None
    assert translate_dataframe is not None
    assert resolve_direction("Hello", "auto")[0] == "en_hi"
