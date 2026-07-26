"""Dataset loaders for reproducible CIFAR-10 experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    root: Path = Path("data/raw")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    download: bool = True


def load_cifar10(config: DatasetConfig = DatasetConfig()) -> tuple[Any, Any]:
    try:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("Install requirements-training.txt to load CIFAR-10.") from exc

    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.CIFAR10(config.root, train=True, download=config.download, transform=train_transform)
    test_dataset = datasets.CIFAR10(config.root, train=False, download=config.download, transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, test_loader
