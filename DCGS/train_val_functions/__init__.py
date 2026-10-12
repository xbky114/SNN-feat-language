from .train_functions import *
from .val_functions import *

def valpool(args):
    if args.task == 'classification':
        if args.mode == 'test_ann':
            return val_ann_classfication
        elif args.mode == 'get_threshold':
            return val_ann_classfication
        elif args.mode == 'test_snn':
            if args.sop:
                return val_snn_classfication_with_sop
            else:
                return val_snn_classfication
    elif args.task == 'object_detection':
        if args.mode == 'test_ann':
            return val_ann_object_detection
        elif args.mode == 'get_threshold':
            return val_ann_object_detection
        elif args.mode == 'test_snn':
            return val_snn_object_detection
    else:
        print("still not support this task for val")
        exit(0)
        