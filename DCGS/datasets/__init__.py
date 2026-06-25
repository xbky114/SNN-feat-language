from .getdataloader import *

def datapool(args):
    if args.dataset == 'cifar10':
        return GetCifar10(args)
    elif args.dataset == 'cifar100':
        return GetCifar100(args)
    elif args.dataset == 'imagenet':
        return GetImageNet(args)
    elif args.dataset == 'coco':
        return GetCOCO(args)
    else:
        print("still not support this dataset")
        exit(0)