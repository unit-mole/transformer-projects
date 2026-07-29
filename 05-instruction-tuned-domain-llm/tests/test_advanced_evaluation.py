from src.advanced_evaluation import EvaluationConfig, score_hallucination_risk, score_instruction_adherence


def test_code_instruction_adherence_rewards_code_format():
    good = score_instruction_adherence(
        "Generate a small Python code example.",
        "```python\nfrom sklearn.model_selection import train_test_split\nX_train, X_test = train_test_split(X)\n```\nUse a fixed seed.",
        "code_example",
    )
    weak = score_instruction_adherence(
        "Generate a small Python code example.",
        "A train test split separates the data.",
        "code_example",
    )
    assert good["adherence_score"] > weak["adherence_score"]
    assert good["adherence_format_score"] == 1.0


def test_hallucination_risk_flags_unsupported_number():
    result = score_hallucination_risk(
        "Explain cross-validation.",
        "Cross-validation always guarantees 99% accuracy.",
        "Cross-validation estimates generalization across folds.",
        0.2,
        EvaluationConfig(),
    )
    assert result["hallucination_risk_flag"]
    assert "unsupported_numeric_claim_review" in result["hallucination_risk_types"]
    assert "low_reference_support" in result["hallucination_risk_types"]
