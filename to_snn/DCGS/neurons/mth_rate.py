import torch
from torch import nn


class MTHRate(nn.Module):
    """Inference equivalent of ANN-to-SNN-DCGS ``relu_neuron.MTH``."""
    def __init__(self, threshold=1.0, num_thresholds=8):
        super().__init__()
        self.register_buffer("threshold", torch.as_tensor(threshold).detach().clone())
        self.num_thresholds = num_thresholds
        self.T = 0
        self.mem = None
        self.thre_shaped = None

    def _levels(self, x):
        th = self.threshold.to(device=x.device, dtype=x.dtype)
        if th.ndim == 1 and x.ndim >= 2 and th.numel() == x.shape[1]:
            th = th.reshape(1, -1, *([1] * (x.ndim - 2)))
        factors = (2 ** torch.arange(self.num_thresholds, device=x.device)).to(x.dtype)
        factors = factors.reshape((self.num_thresholds,) + (1,) * x.ndim)
        return th.unsqueeze(0) / factors

    def forward(self, x):
        if self.T == 0 or self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
            self.thre_shaped = self._levels(x)
        self.mem = self.mem + x
        positive = (self.mem.unsqueeze(0) >= self.thre_shaped * 0.75).to(x.dtype)
        selected = positive.clone()
        if self.num_thresholds > 1:
            selected[1:] -= positive[:-1]
        spike = torch.sum(self.thre_shaped * selected, dim=0)
        self.mem = self.mem - spike
        self.T += 1
        return spike

    def reset_state(self):
        self.T = 0
        self.mem = None
        self.thre_shaped = None
