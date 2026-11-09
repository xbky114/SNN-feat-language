"""Signed multi-threshold ECMT neuron."""

import torch
from torch import nn

try:
    import triton  # noqa: F401
    from ._two_side_triton import step as _fused_step
except Exception:  # pragma: no cover - exercised when Triton is absent/broken
    _fused_step = None


class TwoSide(nn.Module):
    """Stateful signed multi-threshold neuron used by ECMT.

    Positive and negative thresholds are scalar base scales.  At each level
    they are divided by ``gap**level``.  Negative spikes are only allowed to
    retract cumulative positive output of at least that level.
    """

    def __init__(self, positive_threshold=1.0, negative_threshold=1.0,
                 num_thresholds=8, *, gap=2.0, fused=True,
                 threshold=None):
        super().__init__()
        if threshold is not None:
            if positive_threshold != 1.0 or negative_threshold != 1.0:
                raise ValueError("threshold cannot be combined with signed thresholds")
            positive_threshold = negative_threshold = threshold
        if type(num_thresholds) is not int or num_thresholds < 1:
            raise ValueError("num_thresholds must be a positive integer")
        if not isinstance(gap, (int, float)) or not torch.isfinite(torch.tensor(gap)) or gap <= 0:
            raise ValueError("gap must be a positive finite number")
        positive_threshold = self._scalar_threshold(positive_threshold, "positive_threshold")
        negative_threshold = self._scalar_threshold(negative_threshold, "negative_threshold")
        self.register_buffer("positive_threshold", positive_threshold)
        self.register_buffer("negative_threshold", negative_threshold)
        self.num_thresholds = num_thresholds
        self.gap = float(gap)
        self.fused = bool(fused)
        self.T = 0
        self.mem = None
        self.cumulative_out = None
        self._positive_levels = None
        self._negative_levels = None

    @staticmethod
    def _scalar_threshold(value, name):
        tensor = torch.as_tensor(value).detach().clone()
        if tensor.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        if tensor.is_complex():
            raise ValueError(f"{name} must be real")
        tensor = tensor.reshape(()).float()
        if not torch.isfinite(tensor) or tensor < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return tensor

    def _levels(self, base, x):
        # ECMT enumerates the geometric levels from the largest level down to
        # the calibrated base scale.
        exponents = torch.arange(self.num_thresholds - 1, -1, -1,
                                 device=x.device, dtype=x.dtype)
        return (base.to(device=x.device, dtype=x.dtype)
                * torch.pow(torch.as_tensor(self.gap, device=x.device, dtype=x.dtype), exponents)).contiguous()

    def _initialize(self, x):
        self.mem = torch.zeros_like(x)
        self.cumulative_out = torch.zeros_like(x)
        self._positive_levels = self._levels(self.positive_threshold, x)
        self._negative_levels = self._levels(self.negative_threshold, x)
        self.T = 0

    def forward(self, x):
        if not torch.is_floating_point(x):
            raise TypeError("TwoSide expects a floating-point tensor")
        if (self.T == 0 or self.mem is None or self.mem.shape != x.shape
                or self.mem.dtype != x.dtype or self.mem.device != x.device):
            self._initialize(x)
        if self._can_fuse(x):
            spike, self.mem, self.cumulative_out = _fused_step(
                x, self.mem, self.cumulative_out,
                self._positive_levels, self._negative_levels)
        else:
            charged = self.mem + x
            positive = (charged.unsqueeze(0)
                        >= (self._positive_levels - self.positive_threshold.to(x))
                        .reshape((-1,) + (1,) * x.ndim) + self.positive_threshold.to(x) / 2).to(x.dtype)
            negative = ((charged.unsqueeze(0)
                         <= -(self._negative_levels - self.negative_threshold.to(x) / 2)
                         .reshape((-1,) + (1,) * x.ndim))
                        ).to(x.dtype)
            selected_positive = positive.clone()
            selected_negative = negative.clone()
            if self.num_thresholds > 1:
                selected_positive[1:] -= positive[:-1]
                selected_negative[1:] -= negative[:-1]
            positive_sum = torch.sum(self._positive_levels.reshape((-1,) + (1,) * x.ndim)
                                     * selected_positive, dim=0)
            negative_sum = torch.sum(self._negative_levels.reshape((-1,) + (1,) * x.ndim)
                                     * selected_negative, dim=0)
            spike = positive_sum - negative_sum
            self.mem = charged - spike
            self.cumulative_out = self.cumulative_out + spike
        self.T += 1
        return spike

    def _can_fuse(self, x):
        return (self.fused and _fused_step is not None and x.is_cuda
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and x.numel() > 0 and 1 <= self.num_thresholds <= 8
                and x.is_contiguous() and self.mem.is_contiguous()
                and self.cumulative_out.is_contiguous())

    def reset_state(self):
        self.T = 0
        self.mem = None
        self.cumulative_out = None
        self._positive_levels = None
        self._negative_levels = None


TwoSideNeuron = TwoSide
