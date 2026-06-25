import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import MergeTemporalDim, ExpandTemporalDim

class UniformUnpooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(UniformUnpooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

    def forward(self, x):
        N, C, H, W = x.size()
        # 使用重复和reshape来扩展特征图
        x = x.view(N, C, H, 1, W, 1)
        x = x.repeat(1, 1, 1, self.kernel_size, 1, self.kernel_size)
        x = x.view(N, C, H * self.kernel_size, W * self.kernel_size)
        return x.contiguous()

class maxpool_neuron(nn.Module):
    def __init__(self,maxpool,T=8,step_mode='s',coding_type='diff_rate'):
        super(maxpool_neuron,self).__init__()
        self.v = None
        self.maxpool = maxpool
        self.unpool = UniformUnpooling(kernel_size=maxpool.kernel_size,stride=maxpool.stride)
        self.step_mode = step_mode
        self.coding_type = coding_type
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                self.T = -1
            else:
                self.T = 0
        else:
            self.T = T
        if 'diff' in coding_type:
            self.expand = ExpandTemporalDim(T+1)
        else:
            self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        
    def forward(self, x):
        if self.maxpool.kernel_size!=self.maxpool.stride:
            return self.maxpool(x)
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                if self.T==-1:
                    self.bias = x.clone().detach()
                    self.exp_in = x.clone().detach()
                    bias_out = self.maxpool(x)
                    self.exp_out = torch.zeros_like(bias_out)
                    self.T = self.T + 1
                    return torch.zeros_like(bias_out)
                elif self.T==0:
                    self.v = x.clone().detach()
                else:
                    self.v = self.v + x - self.bias + self.exp_in
                    
                self.exp_in = self.exp_in + (x - self.bias)/(self.T+1)
                output = self.maxpool(self.v)
                self.v -= self.unpool(output)
                output = output - self.exp_out
                self.exp_out = self.exp_out + output/(self.T+1)
                self.T = self.T + 1
                return output
            else:
                if self.T==0:
                    self.v = x.clone.detach()
                else:
                    self.v = self.v + x
                output = self.maxpool(self.v)
                self.v -= self.unpool(output)
                self.T = self.T + 1
                return output
            
        else:
            x = self.expand(x)
            output_pot = []
            if 'diff' in self.coding_type:
                bias = x[0].clone().detach()
                bias_out = self.maxpool(x[0])
                output_pot.append(torch.zeros_like(bias_out))
                v = torch.zeros_like(x[0])
                exp_in = x[0].clone().detach()
                exp_out = torch.zeros_like(bias_out)
                for t in range(self.T):
                    v = v + x[t+1] - bias + exp_in 
                    exp_in = exp_in + (x[t+1] - bias)/(t+1)
                    output = self.maxpool(v)
                    v -= self.unpool(output)
                    output = output - exp_out
                    exp_out = exp_out + output/(t+1)
                    output_pot.append(output)
                out = torch.stack(output_pot,dim=0)
                return self.merge(out)
            else:
                v = torch.zeros_like(x[0])
                for t in range(self.T):
                    v = v + x[t] 
                    output = self.maxpool(v)
                    v -= self.unpool(output)
                    output_pot.append(output)
                out = torch.stack(output_pot,dim=0)
                return self.merge(out)
        
    def reset(self):
        self.v = None
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                self.T=-1
            else:
                self.T=0


class exp_comp_neuron(nn.Module):
    def __init__(self,func,T=8,step_mode='s',coding_type='diff_rate'):
        super(exp_comp_neuron,self).__init__()
        self.tot = None
        self.bias = None
        self.func = func
        self.step_mode = step_mode
        self.coding_type = coding_type
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                self.T = -1
            else:
                self.T = 0
        else:
            self.T = T
        if 'diff' in coding_type:
            self.expand = ExpandTemporalDim(T+1)
        else:
            self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
    def forward(self, x):
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                if self.T==-1:
                    self.bias = x.clone().detach()
                    self.last = torch.zeros_like(x)
                    self.exp_in = x.clone().detach()
                    self.T+=1
                    return torch.zeros_like(x)
                else:
                    self.exp_in = self.exp_in + (x - self.bias)/(self.T+1)
                    now = self.func(self.exp_in)
                    output = (self.T+1)*(now - self.last)
                    self.last = now
                self.T+=1
                return output
            else:
                if self.tot==None:
                    last=torch.zeros_like(x)
                    self.tot=x.clone()
                else:
                    last = self.func(self.tot/self.T)*self.T
                    self.tot+=x
                self.T+=1
                now = self.func(self.tot/self.T)*self.T
                return now-last
        else:
            x = self.expand(x)
            output_pot = []
            if 'diff' in self.coding_type:
                last = torch.zeros_like(x[0])
                exp_in = x[0].clone().detach()
                output_pot.append(torch.zeros_like(x[0]))
                for t in range(self.T):
                    exp_in = exp_in + (x[t+1] - x[0])/(t+1)
                    now = self.func(exp_in)
                    output = (t+1)*(now - last)
                    last = now.clone().detach()
                    output_pot.append(output)
                out = torch.stack(output_pot,dim=0)
                return self.merge(out)
            else:
                v = torch.zeros_like(x[0])
                last = torch.zeros_like(x[0])
                for t in range(self.T):
                    v = v + x[t]
                    now = (t+1)*self.func(v/(t+1))
                    output_pot.append(now-last)
                    last = now.clone().detach()
                out = torch.stack(output_pot,dim=0)
                return self.merge(out)
    def reset(self):
        self.tot = None
        self.bias = None
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                self.T=-1
            else:
                self.T=0
        
class AtNeuron(nn.Module):
    def __init__(self,T=8,step_mode='s',coding_type='diff_rate'):
        super(AtNeuron, self).__init__()
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.step_mode = step_mode
        self.coding_type = coding_type
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                self.T = -1
            else:
                self.T = 0
        else:
            self.T = T
        if 'diff' in coding_type:
            self.expand = ExpandTemporalDim(T+1)
        else:
            self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
    def forward(self, x,y):
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                if self.T == -1:
                    self.T+=1
                    return torch.zeros_like(x@y)
                elif self.T == 0:
                    self.tot_a=x
                    self.tot_b=y
                    self.T+=1
                    return x@y
                else:
                    output = x@y/(self.T+1)+x@self.tot_b+self.tot_a@y
                    self.tot_a+=x/(self.T+1)
                    self.tot_b+=y/(self.T+1)
                    self.T+=1
                    return output
            else:
                if self.T == 0:
                    self.tot_a=x
                    self.tot_b=y
                    self.tot_t = x@y
                    self.T = 1
                    return x@y
                else:
                    self.tot_t+= x@y+x@self.tot_b+self.tot_a@y
                    self.tot_a+=x
                    self.tot_b+=y
                    self.T += 1
                    return (x@self.tot_b+self.tot_a@y-x@y)/(self.T-1)-self.tot_t/(self.T*(self.T-1))
        else:
            if 'diff' in self.coding_type:
                x = self.expand(x)
                y = self.expand(y)
                output_pot = []
                x_v = torch.zeros_like(x[0])
                y_v = torch.zeros_like(y[0])
                output_pot.append(torch.zeros_like(x[0]@y[0]))
                for t in range(self.T):
                    if t==0:
                        output = x[t+1]@y[t+1]
                        x_v = x_v + x[t+1]
                        y_v = y_v + y[t+1]
                    else:
                        output = x[t+1]@y[t+1]/(t+1)+x[t+1]@y_v+x_v@y[t+1]
                        x_v = x_v + x[t+1]/(t+1)
                        y_v = y_v + y[t+1]/(t+1)
                    output_pot.append(output)
                out = torch.stack(output_pot,dim=0)
                return self.merge(out)
            else:
                x = self.expand(x)
                y = self.expand(y)
                output_pot = []
                x_v = torch.zeros_like(x[0])
                y_v = torch.zeros_like(y[0])
                t_v = torch.zeros_like(x[0]@y[0])
                for t in range(self.T):
                    tmp_t = t_v.clone().detach()
                    t_v = t_v + x[t]@y[t]+x[t]@y_v+x_v@y[t]
                    x_v = x_v + x[t]
                    y_v = y_v + y[t]
                    if t==0:
                        output = t_v.clone().detach()
                    else:
                        output = t_v/(t+1)-tmp_t/t
                    output_pot.append(output)
                out = torch.stack(output_pot,dim=0)
                return self.merge(out)
            
    def reset(self):# Reset the accumulator
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                self.T=-1
            else:
                self.T=0
                
class MulNeuron(nn.Module):
    def __init__(self,T=8,step_mode='s',coding_type='diff_rate'):
        super(MulNeuron, self).__init__()
        self.tot_a = None
        self.tot_b = None
        self.bias_a = None
        self.bias_b = None
        self.tot_t = None
        self.step_mode = step_mode
        self.coding_type = coding_type
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                self.T = -1
            else:
                self.T = 0
        else:
            self.T = T
        if 'diff' in coding_type:
            self.expand = ExpandTemporalDim(T+1)
        else:
            self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
    def forward(self, x,y):
        if self.step_mode=='s':
            if 'diff' in self.coding_type:#不全对
                if self.T == -1:
                    self.bias_a = x.clone().detach()
                    self.bias_b = y.clone().detach()
                    self.T+=1
                    return torch.zeros_like(x*y)
                elif self.T == 0:
                    self.tot_a=x.clone().detach()
                    self.tot_b=y.clone().detach()
                    self.T+=1
                    return x*y
                else:
                    x = x-self.bias_a
                    y = y-self.bias_b
                    output = x*y/(self.T+1)+x*self.tot_b+self.tot_a*y
                    self.tot_a+=x/(self.T+1)
                    self.tot_b+=y/(self.T+1)
                    self.T+=1
                    return output
            else:
                if self.T == 0:
                    self.tot_a=x
                    self.tot_b=y
                    self.tot_t = x*y
                    self.T+=1
                    return x*y
                else:
                    self.tot_t+= x*y+x*self.tot_b+self.tot_a*y
                    self.tot_a+=x
                    self.tot_b+=y
                    self.T += 1
                    return (x*self.tot_b+self.tot_a*y-x*y)/(self.T-1)-self.tot_t/(self.T*(self.T-1))
        else:
            if 'diff' in self.coding_type:
                x = self.expand(x)
                y = self.expand(y)
                output_pot = []
                x_v = torch.zeros_like(x[0])
                y_v = torch.zeros_like(y[0])
                output_pot.append(torch.zeros_like(x[0]*y[0]))
                
                for t in range(self.T):
                    if t==0:
                        output = x[t+1]*y[t+1]
                        x_v = x_v + x[t+1]
                        y_v = y_v + y[t+1]
                        t_v = x[t+1]*y[t+1]
                    else:
                        output = (x[t+1]-x[0])*(y[t+1]-y[0])/(t+1)+(x[t+1]-x[0])*y_v+x_v*(y[t+1]-y[0])
                        x_v = x_v + (x[t+1]-x[0])/(t+1)
                        y_v = y_v + (y[t+1]-y[0])/(t+1)
                    output_pot.append(output)
                out = torch.stack(output_pot,dim=0)
                return self.merge(out)
            else:
                x = self.expand(x)
                y = self.expand(y)
                output_pot = []
                x_v = torch.zeros_like(x[0])
                y_v = torch.zeros_like(y[0])
                t_v = torch.zeros_like(x[0]*y[0])
                for t in range(self.T):
                    tmp_t = t_v.clone().detach()
                    t_v = t_v + x[t]*y[t]+x[t]*y_v+x_v*y[t]
                    x_v = x_v + x[t]
                    y_v = y_v + y[t]
                    if t==0:
                        output = t_v.clone().detach()
                    else:
                        output = t_v/(t+1)-tmp_t/t
                    output_pot.append(output)
                out = torch.stack(output_pot,dim=0)
                return self.merge(out)
            
    def reset(self):# Reset the accumulator
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.bias_a = None
        self.bias_b = None
        if self.step_mode=='s':
            if 'diff' in self.coding_type:
                self.T=-1
            else:
                self.T=0