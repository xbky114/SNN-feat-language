from .encoder import *
from .decoder import *
import types
from utils import *
import torch
import copy
import numpy as np
def forward_replace(args,model):
    if args.task == 'classification':
        if args.mode == 'test_ann':
            return model
        elif args.mode == 'get_threshold':
            return model
        elif args.mode == 'test_snn':
            model.coding_type = args.coding_type
            model.step_mode = args.step_mode
            if args.coding_type=='rate':
                if args.step_mode=='s':
                    model.T=args.time
                    model.init_forward = model.forward
                    model.forward = types.MethodType(forward_snn_rate_s, model)
                elif args.step_mode=='m':
                    model.T=args.time
                    model.merge = MergeTemporalDim()
                    model.expand = ExpandTemporalDim(model.T)
                    model.init_forward = model.forward
                    model.forward = types.MethodType(forward_snn_rate_m, model)
                else:
                    print("Unexpected step mode")
                return model
            elif args.coding_type=='leaky_rate':
                if args.step_mode=='s':
                    model.T=args.time
                    model.tau = args.tau
                    model.init_forward = model.forward
                    model.forward = types.MethodType(forward_snn_leaky_rate_s, model)
                elif args.step_mode=='m':
                    model.T=args.time
                    model.tau = args.tau
                    model.merge = MergeTemporalDim()
                    model.expand = ExpandTemporalDim(model.T)
                    model.init_forward = model.forward
                    model.forward = types.MethodType(forward_snn_leaky_rate_m, model)
                else:
                    print("Unexpected step mode")
                return model
            elif args.coding_type=='diff_rate':
                if args.step_mode=='s':
                    model.T=args.time
                    model.init_forward = model.forward
                    model.forward = types.MethodType(forward_snn_diff_rate_s, model)
                elif args.step_mode=='m':
                    model.T=args.time
                    model.merge = MergeTemporalDim()
                    model.expand = ExpandTemporalDim(model.T+1)
                    model.init_forward = model.forward
                    model.forward = types.MethodType(forward_snn_diff_rate_m, model)
                else:
                    print("Unexpected step mode")
                return model
            elif args.coding_type=='diff_leaky_rate':
                if args.step_mode=='s':
                    model.T=args.time
                    model.tau = args.tau
                    model.init_forward = model.forward
                    model.forward = types.MethodType(forward_snn_diff_leaky_rate_s, model)
                elif args.step_mode=='m':
                    model.T=args.time
                    model.tau = args.tau
                    model.merge = MergeTemporalDim()
                    model.expand = ExpandTemporalDim(model.T)
                    model.init_forward = model.forward
                    model.forward = types.MethodType(forward_snn_diff_leaky_rate_m, model)
                else:
                    print("Unexpected step mode")
                return model
            else:
                print("Unexpected coding_type")
    elif args.task == 'object_detection':
        if args.mode == 'test_ann':
            return model
        elif args.mode == 'get_threshold':
            return model
        elif args.mode == 'test_snn':
            model.coding_type = args.coding_type
            model.step_mode = args.step_mode
            model.backbone.coding_type = args.coding_type
            model.backbone.step_mode = args.step_mode
            model.head.coding_type = args.coding_type
            model.head.step_mode = args.step_mode
            if args.coding_type=='rate':
                if args.step_mode=='s':
                    pass
                elif args.step_mode=='m':
                    model.backbone.T=args.time
                    model.backbone.merge = MergeTemporalDim()
                    model.backbone.init_forward = model.backbone.forward
                    model.backbone.forward = types.MethodType(forward_snn_rate_m2, model.backbone)
                    model.head.T=args.time
                    model.head.expand = ExpandTemporalDim_dict(model.head.T)
                    model.head.init_forward = model.head.forward
                    model.head.forward = types.MethodType(forward_snn_rate_m3, model.head)
                    model.T=args.time
                    model.forward = model.forward_snn
                else:
                    print("Unexpected step mode")
                return model
            elif args.coding_type=='diff_rate':
                if args.step_mode=='s':
                    pass
                elif args.step_mode=='m':
                    model.backbone.T=args.time
                    model.backbone.merge = MergeTemporalDim()
                    model.backbone.init_forward = model.backbone.forward
                    model.backbone.forward = types.MethodType(forward_snn_diff_rate_m2, model.backbone)
                    model.head.T=args.time
                    model.head.expand = ExpandTemporalDim_dict(model.head.T+1)
                    model.head.init_forward = model.head.forward
                    model.head.forward = types.MethodType(forward_snn_diff_rate_m3, model.head)
                    model.T=args.time
                    model.forward = model.forward_snn
                else:
                    print("Unexpected step mode")
                return model
            else:
                print("Unexpected coding_type")
    else:
        print("still not support this task for val")
        exit(0)
        
def forward_snn_diff_leaky_rate_s(self, x):
    output = []
    output.append(self.init_forward(torch.zeros_like(x)))
    output.append(self.init_forward(x))
    mul = 1
    for i in range(1, self.T):
        mul /= self.tau
        output.append(self.init_forward(torch.zeros_like(x))*mul)
    return decodeoutput(torch.stack(output, dim=0))

def forward_snn_diff_leaky_rate_m(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    out = self.expand(out)
    mul = 1
    for i in range(2,self.T+1):
        mul /= self.tau
        out[i]*=mul
    return decodeoutput(out)        

def forward_snn_diff_rate_s(self, x):
    output = []
    tmp = self.init_forward(torch.zeros_like(x))
    output.append(copy.deepcopy(tmp))
    tmp = self.init_forward(x)
    output.append(copy.deepcopy(tmp))
    for i in range(self.T-1):
        tmp = self.init_forward(torch.zeros_like(x))
        output.append(copy.deepcopy(tmp))
    return decodeoutput(torch.stack(output, dim=0))

def forward_snn_diff_rate_m(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    out = self.expand(out)
    return decodeoutput(out)
        
def forward_snn_leaky_rate_s(self, x):
    output = []
    mul = 1
    for i in range(self.T):
        output.append(self.init_forward(x/mul)*mul)
        mul /= self.tau
    return torch.stack(output, dim=0)

def forward_snn_leaky_rate_m(self, x):
    x = add_dimention(x, self.T)
    for i in range(1,self.T):
        x[i]=x[i-1]/self.tau
    x = self.merge(x)
    out = self.init_forward(x)
    out = self.expand(out)
    mul = 1
    for i in range(self.T):
        out[i]*=mul
        mul /= self.tau
    return out

def forward_snn_rate_s(self, x):
    output = []
    for i in range(self.T):
        tmp = self.init_forward(copy.deepcopy(x))
        output.append(copy.deepcopy(tmp))
    return torch.stack(output, dim=0)

def forward_snn_rate_m(self, x):
    x = add_dimention(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    out = self.expand(out)
    return out

def add_dimention(x, T):
    x.unsqueeze_(0)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

def add_dimention_diff(x, T):
    x.unsqueeze_(0)
    x = x.repeat(T+1, 1, 1, 1, 1)
    x[0] = 0 
    x[2:] = 0
    return x

def decodeoutput(x):
    out = torch.zeros_like(x[1:])
    T = x.shape[0]-1
    exp_in = x[0].clone().detach()
    for t in range(T):
        out[t]= exp_in + x[t+1] - x[0]
        exp_in = exp_in + (x[t+1] - x[0])/(t+1)
    return out


def forward_snn_rate_m2(self, x):
    x = add_dimention(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    return out

def forward_snn_rate_m3(self, x):
    out = self.init_forward(x)
    out = self.expand(out)
    return out

def forward_snn_diff_rate_m2(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    return out

def forward_snn_diff_rate_m3(self, x):
    out = self.init_forward(x)
    out = self.expand(out)
    return {key:decodeoutput(out[key]) for key in out}