import torch
from torch import nn

try:
    import triton
except ImportError:
    _fused_step = None
else:
    from ._mth_triton import step as _fused_step


class MTH(nn.Module):
    """Input-dtype single-step dynamics based on ``relu_neuron.MTH_with_neg``.

    Negative spikes can retract earlier positive output, but cannot make
    cumulative output negative. This is not the ``MTH_diff`` state update.
    """
    def __init__(self, threshold=1.0, num_thresholds=8, *, fused=True):
        super().__init__()
        self.register_buffer("threshold", torch.as_tensor(threshold).detach().clone())
        self.num_thresholds = num_thresholds
        self.fused = fused
        self.T = 0
        self.mem = None
        self.thre_shaped = None
        self.cumulative_out = None

    def _levels(self, x):
        th = self.threshold.to(device=x.device, dtype=x.dtype)
        if th.ndim == 1 and x.ndim >= 2 and th.numel() == x.shape[1]:
            th = th.reshape(1, -1, *([1] * (x.ndim - 2)))
        factors = 2 ** torch.arange(self.num_thresholds, device=x.device)
        factors = factors.reshape((self.num_thresholds,) + (1,) * x.ndim)
        return th.unsqueeze(0) / factors

    def forward(self, x):
        if (self.T == 0 or self.mem is None or self.mem.shape != x.shape
                or self.mem.dtype != x.dtype or self.mem.device != x.device):
            self.mem = torch.zeros_like(x)
            self.thre_shaped = self._levels(x)
            self.cumulative_out = torch.zeros_like(x)
        if self._can_fuse(x):
            spike, self.mem, self.cumulative_out = _fused_step(
                x, self.mem, self.cumulative_out, self.thre_shaped)
            self.T += 1
            return spike
        self.mem = self.mem + x
        positive = (self.mem.unsqueeze(0) >= self.thre_shaped * 0.75).to(dtype=x.dtype)
        negative = ((self.mem.unsqueeze(0) <= -self.thre_shaped * 0.75).to(dtype=x.dtype)
                    * (self.cumulative_out.unsqueeze(0) >= self.thre_shaped).to(dtype=x.dtype))
        selected_positive = positive.clone()
        selected_negative = negative.clone()
        if self.num_thresholds > 1:
            selected_positive[1:] -= positive[:-1]
            selected_negative[1:] -= negative[:-1]
        spike = (torch.sum(self.thre_shaped * selected_positive, dim=0)
                 - torch.sum(self.thre_shaped * selected_negative, dim=0))
        self.mem = self.mem - spike
        self.cumulative_out = self.cumulative_out + spike
        self.T += 1
        return spike

    def _can_fuse(self, x):
        return (self.fused and _fused_step is not None and x.is_cuda
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and x.numel() > 0 and x.ndim <= 4 and 1 <= self.num_thresholds <= 8
                and x.is_contiguous() and self.mem.is_contiguous()
                and self.cumulative_out.is_contiguous()
                and not (torch.is_grad_enabled() and any(
                    value.requires_grad for value in
                    (x, self.mem, self.cumulative_out, self.thre_shaped))))

    def reset_state(self):
        self.T = 0
        self.mem = None
        self.thre_shaped = None
        self.cumulative_out = None
