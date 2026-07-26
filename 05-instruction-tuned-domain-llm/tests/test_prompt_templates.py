from src.prompt_templates import format_prompt


def test_prompt_contains_required_sections():
    prompt = format_prompt("Explain recall", "Use a defect example")
    assert "Instruction:" in prompt
    assert "Input:" in prompt
    assert prompt.endswith("Response:")
