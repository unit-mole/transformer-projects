from src.response_generation import GenerationSettings


def test_generation_defaults_are_bounded():
    settings = GenerationSettings()
    assert 1 <= settings.max_new_tokens <= 512
    assert 0 <= settings.temperature <= 2
    assert 0 < settings.top_p <= 1
