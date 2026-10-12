import torch
from torch import nn
class IFRate(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__(); self.register_buffer('threshold', torch.as_tensor(threshold)); self.v=None
    def forward(self,x):
        th=self.threshold.to(device=x.device,dtype=x.dtype); self.v = torch.zeros_like(x)+0.5*th if self.v is None or self.v.shape!=x.shape else self.v
        self.v = self.v + x; spikes=(self.v>=th).to(x.dtype)*th; self.v=self.v-spikes; return spikes
    def reset_state(self): self.v=None
