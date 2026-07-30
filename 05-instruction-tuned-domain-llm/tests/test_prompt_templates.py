from src.prompt_templates import build_training_prompt


def test_prompt_contains_shared_sections():
    prompt = build_training_prompt("Explain recall.", "Use a defect example.", "Metric explanation")
    assert "System:" in prompt
    assert "Instruction: Explain recall." in prompt
    assert "Input: Use a defect example." in prompt
    assert prompt.endswith("Response:")
