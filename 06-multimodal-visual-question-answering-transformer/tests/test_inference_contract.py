from types import SimpleNamespace
from PIL import Image
from vqa.inference_pipeline import VQAResult

def test_result_contract_is_serializable():
    result = VQAResult(
        answer="red",
        confidence_proxy=0.7,
        confidence_label="medium",
        confidence_margin=0.2,
        question_type="color",
        answer_type="other",
        latency_seconds=0.5,
        model_id="test/model",
        device="cpu",
        image={"width": 10, "height": 10},
    )
    assert result.to_dict()["answer"] == "red"
