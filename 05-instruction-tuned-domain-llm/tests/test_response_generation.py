from src.response_generation import generation_kwargs


def test_deterministic_generation_when_temperature_is_zero():
    kwargs = generation_kwargs(128, 0.0, 0.9, 1.1)
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 4


def test_sampling_generation_when_temperature_is_positive():
    kwargs = generation_kwargs(128, 0.4, 0.85, 1.1)
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.4
    assert kwargs["top_p"] == 0.85
