"""Inference-only fused MTH update; imported only when Triton is available."""

import torch
import triton
import triton.language as tl


@triton.jit
def _step_kernel(
    X, MEM, CUMULATIVE, LEVELS, SPIKE, NEXT_MEM, NEXT_CUMULATIVE,
    N: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
    LEVEL_STRIDE: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    dtype = X.dtype.element_ty
    # Preserve eager's rounding after each materialized low-precision op.
    x = tl.load(X + offsets, valid, other=0).to(tl.float32)
    mem = tl.load(MEM + offsets, valid, other=0).to(tl.float32)
    cumulative = tl.load(CUMULATIVE + offsets, valid, other=0).to(tl.float32)
    charged = (mem + x).to(dtype).to(tl.float32)

    threshold_offset = ((offsets // (D1 * D2 * D3)) * S0
                        + (offsets // (D2 * D3) % D1) * S1
                        + (offsets // D3 % D2) * S2
                        + (offsets % D3) * S3)

    prev_positive = tl.full((BLOCK,), 0, tl.float32)
    prev_negative = tl.full((BLOCK,), 0, tl.float32)
    positive_sum = tl.full((BLOCK,), 0, tl.float32)
    negative_sum = tl.full((BLOCK,), 0, tl.float32)
    for level in tl.static_range(K):
        threshold = tl.load(LEVELS + level * LEVEL_STRIDE + threshold_offset,
                            valid, other=0).to(tl.float32)
        boundary = (threshold * 0.75).to(dtype).to(tl.float32)
        positive = (charged >= boundary).to(tl.float32)
        negative = ((charged <= -boundary) & (cumulative >= threshold)).to(tl.float32)
        positive_sum += threshold * (positive - prev_positive)
        negative_sum += threshold * (negative - prev_negative)
        prev_positive = positive
        prev_negative = negative

    # torch.sum accumulates in FP32 but returns the input dtype. Each sum
    # rounds separately before the subtraction, as in the eager implementation.
    positive_sum = positive_sum.to(dtype).to(tl.float32)
    negative_sum = negative_sum.to(dtype).to(tl.float32)
    spike = (positive_sum - negative_sum).to(dtype).to(tl.float32)
    tl.store(SPIKE + offsets, spike, valid)
    tl.store(NEXT_MEM + offsets, charged - spike, valid)
    tl.store(NEXT_CUMULATIVE + offsets, cumulative + spike, valid)


def step(x, mem, cumulative, levels):
    """Return spike and new states without modifying any input tensors."""
    spike = torch.empty_like(x)
    next_mem = torch.empty_like(x)
    next_cumulative = torch.empty_like(x)
    expanded = levels.expand((levels.shape[0],) + tuple(x.shape))
    shape = (1,) * (4 - x.ndim) + tuple(x.shape)
    strides = (0,) * (4 - x.ndim) + tuple(expanded.stride()[1:])
    with torch.cuda.device(x.device):
        _step_kernel[(triton.cdiv(x.numel(), 256),)](
            x, mem, cumulative, levels, spike, next_mem, next_cumulative,
            x.numel(), *shape[1:], *strides,
            expanded.stride(0), levels.shape[0], 256,
            num_warps=4,
        )
    return spike, next_mem, next_cumulative
