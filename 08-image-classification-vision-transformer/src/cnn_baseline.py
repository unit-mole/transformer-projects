"""Matched CNN baseline for comparison experiments."""
from __future__ import annotations


def build_resnet18(num_classes: int = 10, pretrained: bool = False):
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError("Install requirements-training.txt to build the baseline.") from exc
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = __import__("torch").nn.Linear(model.fc.in_features, num_classes)
    return model
