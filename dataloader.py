"""ImageNet dataloaders used by ANN evaluation and SNN calibration."""
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

def resolve_class_names(dataset):
    """Resolve ImageNet folder IDs to readable names in label-index order."""
    if not any(re.fullmatch(r'n\d{8}', name) for name in dataset.classes):
        return dataset.classes
    from scipy.io import loadmat
    root = Path(dataset.root).resolve()
    for parent in (root, *root.parents):
        metadata = parent / 'ILSVRC2012_devkit_t12/data/meta.mat'
        if metadata.is_file():
            synsets = loadmat(metadata, squeeze_me=True)['synsets']
            names = {str(s['WNID']): str(s['words']).split(',')[0] for s in synsets}
            return [names[name] for name in dataset.classes]
    raise FileNotFoundError('ImageNet WordNet folder IDs need the devkit: '
                            'place ILSVRC2012_devkit_t12 beside the dataset directory')

def _transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130211, 0.27577711)),
    ])

def imagenet_dataloaders(root: str | Path, *, batch_size: int = 32,
                         calibration_size: Optional[int] = 1024,
                         image_size: int = 224, num_workers: int = 4):
    """Return ``(evaluation_loader, calibration_loader)`` for ImageFolder."""
    dataset = datasets.ImageFolder(str(root), transform=_transform(image_size))
    kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())
    evaluation = DataLoader(dataset, **kwargs)
    if calibration_size is None or calibration_size >= len(dataset):
        calibration_dataset = dataset
    else:
        calibration_dataset = Subset(dataset, range(max(0, calibration_size)))
    return evaluation, DataLoader(_ImageOnly(calibration_dataset), **kwargs)

def get_dataloaders(*args, **kwargs):
    return imagenet_dataloaders(*args, **kwargs)

build_dataloaders = imagenet_dataloaders

def imagenet_dataloader(root, **kwargs):
    return imagenet_dataloaders(root, **kwargs)[0]

def calibration_dataloader(root, **kwargs):
    return imagenet_dataloaders(root, **kwargs)[1]
