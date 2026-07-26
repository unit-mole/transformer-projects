from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from .confidence_scoring import confidence_from_logits
from .image_preprocessing import image_summary, load_and_validate_image
from .model_loader import LoadedVilt, load_vilt_model
from .question_preprocessing import (
    classify_answer_type,
    classify_question_type,
    preprocess_question,
)

@dataclass
class VQAResult:
    answer: str
    confidence_proxy: float
    confidence_label: str
    confidence_margin: float
    question_type: str
    answer_type: str
    latency_seconds: float
    model_id: str
    device: str
    image: dict
    disclaimer: str = (
        "Confidence is an uncalibrated model-based proxy and does not guarantee correctness."
    )

    def to_dict(self) -> dict:
        return asdict(self)

class VQAInferencePipeline:
    def __init__(self, loaded: LoadedVilt | None = None, **loader_kwargs: Any) -> None:
        self.loaded = loaded or load_vilt_model(**loader_kwargs)

    def predict(self, image_input: Any, question: str) -> VQAResult:
        import torch

        image = load_and_validate_image(image_input)
        clean_question = preprocess_question(question)
        started = perf_counter()
        inputs = self.loaded.processor(
            images=image,
            text=clean_question,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.loaded.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.loaded.model(**inputs)
        logits = outputs.logits[0].detach().float().cpu().tolist()
        best_id = int(max(range(len(logits)), key=logits.__getitem__))
        answer = self.loaded.model.config.id2label.get(best_id, str(best_id))
        proxy = confidence_from_logits(logits)
        latency = perf_counter() - started
        return VQAResult(
            answer=answer,
            confidence_proxy=round(proxy.top_probability, 6),
            confidence_label=proxy.label,
            confidence_margin=round(proxy.margin, 6),
            question_type=classify_question_type(clean_question),
            answer_type=classify_answer_type(answer),
            latency_seconds=round(latency, 4),
            model_id=self.loaded.model_id,
            device=self.loaded.device,
            image=image_summary(image),
        )
