"""Inference-only fused update for :class:`TwoSide`."""

import torch
import triton
import triton.language as tl


@triton.jit
def _step_kernel(
    x_ptr, mem_ptr, cumulative_ptr, pos_ptr, neg_ptr,
    spike_ptr, next_mem_ptr, next_cumulative_ptr,
    n_elements, k: tl.constexpr, block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    dtype = x_ptr.dtype.element_ty
    x = tl.load(x_ptr + offsets, mask, other=0).to(tl.float32)
    mem = tl.load(mem_ptr + offsets, mask, other=0).to(tl.float32)
    cumulative = tl.load(cumulative_ptr + offsets, mask, other=0).to(tl.float32)
    charged = (mem + x).to(dtype).to(tl.float32)
    previous_positive = tl.zeros((block,), dtype=tl.float32)
    previous_negative = tl.zeros((block,), dtype=tl.float32)
    positive_sum = tl.zeros((block,), dtype=tl.float32)
    negative_sum = tl.zeros((block,), dtype=tl.float32)
    for level in tl.static_range(k):
        positive_threshold = tl.load(pos_ptr + level).to(tl.float32)
        negative_threshold = tl.load(neg_ptr + level).to(tl.float32)
        positive_base = tl.load(pos_ptr + (k - 1)).to(tl.float32)
        negative_base = tl.load(neg_ptr + (k - 1)).to(tl.float32)
        positive_boundary = (positive_threshold - positive_base * 0.5).to(dtype).to(tl.float32)
        negative_boundary = (negative_threshold - negative_base * 0.5).to(dtype).to(tl.float32)
        positive = (charged >= positive_boundary).to(tl.float32)
        negative = (charged <= -negative_boundary).to(tl.float32)
        positive_sum += positive_threshold * (positive - previous_positive)
        negative_sum += negative_threshold * (negative - previous_negative)
        previous_positive = positive
        previous_negative = negative
    positive_sum = positive_sum.to(dtype).to(tl.float32)
    negative_sum = negative_sum.to(dtype).to(tl.float32)
    spike = (positive_sum - negative_sum).to(dtype).to(tl.float32)
    tl.store(spike_ptr + offsets, spike, mask)
    tl.store(next_mem_ptr + offsets, charged - spike, mask)
    tl.store(next_cumulative_ptr + offsets, cumulative + spike, mask)


def step(x, mem, cumulative, positive_levels, negative_levels):
    """Return one fused update without modifying input state tensors."""
    spike = torch.empty_like(x)
    next_mem = torch.empty_like(x)
    next_cumulative = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _step_kernel[(triton.cdiv(x.numel(), 256),)](
            x, mem, cumulative, positive_levels, negative_levels,
            spike, next_mem, next_cumulative, x.numel(),
            positive_levels.numel(), 256, num_warps=4,
        )
    return spike, next_mem, next_cumulative
