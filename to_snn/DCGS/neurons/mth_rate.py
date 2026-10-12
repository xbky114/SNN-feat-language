import torch
from torch import nn
class MTHRate(nn.Module):
    def __init__(self, threshold=1.0, num_thresholds=1):
        super().__init__(); self.register_buffer('threshold', torch.as_tensor(threshold)); self.num_thresholds=num_thresholds; self.v=None
    def forward(self,x):
        th=self.threshold.to(device=x.device,dtype=x.dtype); self.v=torch.zeros_like(x) if self.v is None or self.v.shape!=x.shape else self.v; self.v=self.v+x
        if th.ndim == 1 and x.ndim >= 3 and th.numel() == x.shape[1]:
            th = th.reshape(1, -1, *([1] * (x.ndim - 2)))
        n=torch.floor(torch.clamp(self.v/th,min=0,max=self.num_thresholds)); self.v=self.v-n*th; return n*th
    def reset_state(self): self.v=None
