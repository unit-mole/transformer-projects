"""Vision Transformer loading and classifier-head adaptation."""
from __future__ import annotations

from typing import Any

from .class_mapping import build_mappings

DEFAULT_MODEL = "facebook/deit-tiny-patch16-224"


def load_vit_model(model_name: str = DEFAULT_MODEL, class_names: list[str] | None = None) -> tuple[Any, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError("Install requirements-training.txt to load Transformers models.") from exc

    processor = AutoImageProcessor.from_pretrained(model_name)
    kwargs: dict[str, Any] = {}
    if class_names:
        id2label, label2id = build_mappings(class_names)
        kwargs.update(num_labels=len(class_names), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    model = AutoModelForImageClassification.from_pretrained(model_name, **kwargs)
    return processor, model


def count_parameters(model: Any, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(int(p.numel()) for p in parameters)
