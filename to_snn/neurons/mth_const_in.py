"""Stateless multi-step MTH for constant inputs and independent thresholds."""

import torch

from .mth import MTH

try:
    import triton
except ImportError:
    _run = None
else:
    from ._mth_const_in_triton import run as _run


@torch.no_grad()
def mth_const_in(x, thresholds, time_steps, num_thresholds=4,
                 coding="rate_coding", *, fused=True):
    """Decode constant 1-D input for each scalar threshold, starting at rest.

    Returns [candidate, activation] in x.dtype. Each call starts from zero
    membrane and cumulative output; no trajectory or state is retained.
    Zero-threshold candidates produce zero, matching MMSE's search convention.
    Unsupported devices/dtypes/threshold counts use the single-step MTH.
    """
    if x.ndim != 1 or not x.is_floating_point():
        raise ValueError("x must be a one-dimensional floating-point tensor")
    if type(time_steps) is not int or time_steps < 1:
        raise ValueError("time_steps must be a positive integer")
    if type(num_thresholds) is not int or num_thresholds < 1:
        raise ValueError("num_thresholds must be a positive integer")
    if coding not in ("rate_coding", "differential_coding"):
        raise ValueError(f"unsupported coding: {coding!r}")
    thresholds = torch.as_tensor(thresholds, device=x.device).reshape(-1, 1)
    zero = thresholds == 0
    neuron = MTH(thresholds, num_thresholds, fused=False)
    inputs = x.unsqueeze(0).expand(thresholds.numel(), -1)
    if (fused and _run is not None and x.is_cuda and x.numel()
            and thresholds.numel() and num_thresholds <= 8
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)):
        levels = neuron._levels(inputs).contiguous()
        return _run(x.contiguous(), levels, zero, time_steps, coding)
    accumulated = None
    for t in range(1, time_steps + 1):
        spike = neuron(inputs).masked_fill(zero, 0)
        term = spike / t if coding == "differential_coding" else spike
        if term.dtype in (torch.float16, torch.bfloat16):
            term = term.float()
        accumulated = term if accumulated is None else accumulated + term
    if coding == "rate_coding":
        accumulated = accumulated / time_steps
    return accumulated.to(spike.dtype)
