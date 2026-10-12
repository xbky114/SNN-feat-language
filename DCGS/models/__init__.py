from .ResNet import *
from .VGG import *
from .ViT import *
from .Eva import *
from .det_models import *

def modelpool(args):
    if args.task == 'classification':
        if args.dataset == 'imagenet':
            num_classes = 1000
        elif args.dataset == 'cifar100':
            num_classes = 100
        elif args.dataset == 'cifar10':
            num_classes = 10
        else:
            print("still not support this dataset")
            exit(0)
        if args.model_name == 'vgg16_bn':
            return vgg16(num_classes=num_classes)
        elif args.model_name == 'vgg16_wobn':
            return vgg16_wobn(num_classes=num_classes)
        elif args.model_name == 'vgg19_bn':
            return vgg19(num_classes=num_classes)
        elif args.model_name == 'resnet18':
            return resnet18(num_classes=num_classes)
        elif args.model_name == 'resnet20':
            return resnet20(num_classes=num_classes)
        elif args.model_name == 'resnet34':
            return resnet34(num_classes=num_classes)
        elif args.model_name == 'resnet50':
            return resnet34(num_classes=num_classes)
        elif args.model_name == 'resnet152':
            return resnet34(num_classes=num_classes)
        elif args.model_name == 'vit_small':
            return vit_small_patch16_224(num_classes=num_classes)
        elif args.model_name == 'vit_base':
            return vit_base_patch16_224(num_classes=num_classes)
        elif args.model_name == 'vit_large':
            return vit_large_patch16_224(num_classes=num_classes)
        elif args.model_name == 'eva02_tiny':
            return eva02_tiny_patch14_336(num_classes=num_classes)
        elif args.model_name == 'eva02_small':
            return eva02_small_patch14_336(num_classes=num_classes)
        elif args.model_name == 'eva02_base':
            return eva02_base_patch14_448(num_classes=num_classes)
        elif args.model_name == 'eva02_large':
            return eva02_large_patch14_448(num_classes=num_classes)
        else:
            print("still not support this model")
            exit(0)
    elif args.task == 'object_detection':
        if args.dataset == 'coco':
            num_classes = 91
        else:
            print("error dataset")
        if args.model_name == 'fcos_resnet50_fpn':
            return fcos_resnet50_fpn(num_classes=num_classes)
        elif args.model_name == 'retinanet_resnet50_fpn':
            return retinanet_resnet50_fpn(num_classes=num_classes)
        elif args.model_name == 'retinanet_resnet50_fpn_v2':
            return retinanet_resnet50_fpn_v2(num_classes=num_classes)
    else:
        print("still not support this task")
        exit(0)
        