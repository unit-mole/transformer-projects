"""Minimal reusable PyTorch training loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def train_one_epoch(model: Any, loader: Any, optimizer: Any, device: Any) -> EpochResult:
    import torch
    model.train()
    total_loss = total_correct = total_examples = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(pixel_values=inputs) if "pixel_values" in getattr(model.forward, "__annotations__", {}) else model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        batch = labels.size(0)
        total_loss += float(loss.item()) * batch
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch
    return EpochResult(total_loss / max(total_examples, 1), total_correct / max(total_examples, 1))
