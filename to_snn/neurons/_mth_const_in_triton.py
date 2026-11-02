"""Temporal fusion for constant-input MTH; separate from single-step MTH."""

import torch
import triton
import triton.language as tl
from triton.language import math as libdevice


@triton.jit
def _constant_kernel(X, LEVELS, ZERO, OUTPUT,
                     N: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                     T: tl.constexpr, DIFFERENTIAL: tl.constexpr,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    candidate = tl.program_id(1)
    valid = offsets < N
    dtype = X.dtype.element_ty
    x = tl.load(X + offsets, valid, other=0).to(tl.float32)
    zero = tl.load(ZERO + candidate)
    base_threshold = tl.load(LEVELS + candidate).to(tl.float32)
    mem = tl.full((BLOCK,), 0, tl.float32)
    cumulative = tl.full((BLOCK,), 0, tl.float32)
    accumulated = tl.full((BLOCK,), 0, tl.float32)
    for t in range(1, T + 1):
        charged = (mem + x).to(dtype).to(tl.float32)
        prev_positive = tl.full((BLOCK,), 0, tl.float32)
        prev_negative = tl.full((BLOCK,), 0, tl.float32)
        positive_sum = tl.full((BLOCK,), 0, tl.float32)
        negative_sum = tl.full((BLOCK,), 0, tl.float32)
        if (base_threshold >= 0.) & (base_threshold < float('inf')):
            # For nonnegative descending levels, the eager adjacent-mask
            # difference selects exactly the largest firing threshold. Select
            # it directly, avoiding per-level mask subtraction and reduction.
            for level in tl.static_range(K - 1, -1, -1):
                threshold = tl.load(LEVELS + level * C + candidate).to(tl.float32)
                boundary = (threshold * 0.75).to(dtype).to(tl.float32)
                positive_sum = tl.where(charged >= boundary, threshold, positive_sum)
                negative_sum = tl.where((charged <= -boundary) & (cumulative >= threshold),
                                        threshold, negative_sum)
        else:
            for level in tl.static_range(K):
                threshold = tl.load(LEVELS + level * C + candidate).to(tl.float32)
                boundary = (threshold * 0.75).to(dtype).to(tl.float32)
                positive = (charged >= boundary).to(tl.float32)
                negative = ((charged <= -boundary) & (cumulative >= threshold)).to(tl.float32)
                positive_sum += threshold * (positive - prev_positive)
                negative_sum += threshold * (negative - prev_negative)
                prev_positive = positive
                prev_negative = negative
        positive_sum = positive_sum.to(dtype).to(tl.float32)
        negative_sum = negative_sum.to(dtype).to(tl.float32)
        spike = (positive_sum - negative_sum).to(dtype).to(tl.float32)
        mem = (charged - spike).to(dtype).to(tl.float32)
        cumulative = (cumulative + spike).to(dtype).to(tl.float32)
        spike = tl.where(zero, 0., spike)
        if DIFFERENTIAL:
            # PyTorch scalar division multiplies by the FP32 reciprocal.
            reciprocal = libdevice.div_rn(1., t.to(tl.float32))
            term = libdevice.mul_rn(spike, reciprocal)
            term = term.to(dtype).to(tl.float32)
        else:
            term = spike
        accumulated = accumulated + term
    if not DIFFERENTIAL:
        accumulated = accumulated * (1. / T)
    tl.store(OUTPUT + candidate * N + offsets, accumulated, valid)


def run(x, levels, zero, time_steps, coding):
    candidates = levels.shape[1]
    output = torch.empty((candidates, x.numel()), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        _constant_kernel[(triton.cdiv(x.numel(), 128), candidates)](
            x, levels, zero, output, x.numel(), candidates, levels.shape[0],
            time_steps, coding == "differential_coding", 128, num_warps=4)
    return output
