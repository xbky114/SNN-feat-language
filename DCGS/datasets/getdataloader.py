from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
from datasets.augment import Cutout, CIFAR10Policy

# your own data dir
DIR = {'CIFAR10': '../data', 'CIFAR100': '../data', 'ImageNet': '../data/', 'COCO': '../data/', }

def GetCifar10(args):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(args.dataset_path, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(args.dataset_path, train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

def GetCifar100(args):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = datasets.CIFAR100(args.dataset_path, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(args.dataset_path, train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader


def create_dataloader(dataset, batch_size, shuffle, num_workers, distributed):
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler, pin_memory=torch.cuda.is_available())
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
def GetImageNet(args):
    if 'resnet' in args.model_name or 'vgg' in args.model_name:
        trans = transforms.Compose([
                    transforms.Resize(size=235, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(size=(224, 224)),
                    transforms.ToTensor(),  # 使用 timm 的 MaybeToTensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    elif 'vit' in args.model_name:
        trans = transforms.Compose([
                    transforms.Resize(size=248, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
    elif args.model_name in ['eva02_tiny','eva02_small']: ##'eva'
        trans = transforms.Compose([
                    transforms.Resize(size=336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(size=(336, 336)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
                ])
    elif args.model_name in ['eva02_base','eva02_large']: ##'eva'
        trans = transforms.Compose([
                    transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(size=(448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
                ])
    
    if args.mode in ['train_snn']:
        train_data = datasets.ImageFolder(root=os.path.join(args.dataset_path, 'train'), transform=trans)
        train_dataloader = create_dataloader(train_data, args.batchsize, shuffle=True, num_workers=8, distributed=args.distributed)
    else:
        train_dataloader = None

    test_data = datasets.ImageFolder(root=os.path.join(args.dataset_path, 'val'), transform=trans)
    test_dataloader = create_dataloader(test_data, args.batchsize, shuffle=False, num_workers=2, distributed=args.distributed)

    
    return train_dataloader, test_dataloader

# resnet/vgg
# Compose(
#     Resize(size=235, interpolation=bicubic, max_size=None, antialias=warn)
#     CenterCrop(size=(224, 224))
#     MaybeToTensor()
#     Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
# )

# vit
# Compose(
#     Resize(size=248, interpolation=bicubic, max_size=None, antialias=warn)
#     CenterCrop(size=(224, 224))
#     MaybeToTensor()
#     Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
# )

# eva
# Compose(
#     Resize(size=336, interpolation=bicubic, max_size=None, antialias=warn)
#     CenterCrop(size=(336, 336))
#     MaybeToTensor()
#     Normalize(mean=tensor([0.4815, 0.4578, 0.4082]), std=tensor([0.2686, 0.2613, 0.2758]))
# )

from torchvision.datasets import CocoDetection
class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target
    
def GetCOCO(args):
    transform = ComposeTransforms([
        transforms.ToTensor(),
    ])
    train_loader = None
    val_dataset = CocoDetection(
        root=os.path.join(args.dataset_path, 'COCO/val2017'),
        annFile=os.path.join(args.dataset_path, 'COCO/annotations/instances_val2017.json'),
        transforms=transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    return train_loader, val_loader