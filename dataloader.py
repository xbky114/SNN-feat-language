"""ImageNet dataloaders used by ANN evaluation and SNN calibration.
exports: calibration_dataloader (image only) and imagenet_dataloader
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import re

class _ImageOnly:
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image


def imagenet_transform(image_size: int = 224):
    """Return the CLIP/ImageNet validation preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130211, 0.27577711)),
    ])

def _loader_kwargs(batch_size: int, num_workers: int) -> dict:
    return dict(batch_size=batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=torch.cuda.is_available())


def imagenet_dataloader(root: str | Path, *, batch_size: int = 32,
                        image_size: int = 224, num_workers: int = 4):
    """Build the ordinary ImageNet ``ImageFolder`` loader (images, labels)."""
    dataset = datasets.ImageFolder(str(root), transform=imagenet_transform(image_size))
    return DataLoader(dataset, **_loader_kwargs(batch_size, num_workers))


def calibration_dataloader(root: str | Path, *, batch_size: int = 32,
                          calibration_size: Optional[int] = 1024,
                          image_size: int = 224, num_workers: int = 4):
    """Build an image-only loader for calibration.

    Samples are taken in ImageFolder order, making calibration deterministic.
    ``calibration_size=None`` uses the complete dataset.
    """
    dataset = datasets.ImageFolder(str(root), transform=imagenet_transform(image_size))
    if calibration_size is not None and calibration_size < len(dataset):
        dataset = Subset(dataset, range(max(0, calibration_size)))
    return DataLoader(_ImageOnly(dataset), **_loader_kwargs(batch_size, num_workers))

