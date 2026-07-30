from src.inference_pipeline import InstructionAssistant


def test_scope_guard_avoids_model_loading():
    assistant = InstructionAssistant()
    result = assistant.generate("Give me medical treatment advice.")
    assert result["model_mode"] == "scope_guard"
    assert assistant.loaded is False


def test_empty_prompt_avoids_model_loading():
    assistant = InstructionAssistant()
    result = assistant.generate("")
    assert result["model_mode"] == "not_loaded"
    assert assistant.loaded is False
