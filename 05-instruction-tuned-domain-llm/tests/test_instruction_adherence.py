from src.instruction_adherence import evaluate_instruction_adherence


def test_comparison_response_receives_format_credit():
    result = evaluate_instruction_adherence(
        "Compare precision and recall.",
        "Precision measures correct positive predictions, while recall measures captured actual positives in a classification model.",
    )
    assert result["follows_requested_format"] is True
    assert result["stays_in_ml_ds_scope"] is True
