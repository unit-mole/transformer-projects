from src.inference_pipeline import validate_user_prompt


def test_empty_prompt_is_blocked():
    ok, _ = validate_user_prompt("   ")
    assert not ok


def test_ml_prompt_is_allowed():
    ok, message = validate_user_prompt("Explain logistic regression")
    assert ok
    assert message == "Explain logistic regression"
