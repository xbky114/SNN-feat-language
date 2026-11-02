"""Bounded candidate batching and disk prefetch for independent MMSE searches."""
from concurrent.futures import ThreadPoolExecutor
import time
import math

import torch

from to_snn.neurons import IF, MTH, mth_const_in


def _prefetch(paths):
    """Keep at most one additional activation file in host memory."""
    def read(path):
        return torch.load(path, map_location="cpu", weights_only=True).reshape(-1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        iterator = iter(paths)
        path = next(iterator, None)
        pending = pool.submit(read, path) if path is not None else None
        while pending is not None:
            activation = pending.result()
            path = next(iterator, None)
            pending = pool.submit(read, path) if path is not None else None
            yield activation


def _device_chunks(paths, chunk_size, device):
    chunks = ((chunk, file_index == len(paths) - 1 and
               offset + chunk_size >= activation.numel())
              for file_index, activation in enumerate(_prefetch(paths))
              for offset in range(0, activation.numel(), chunk_size)
              for chunk in (activation[offset:offset + chunk_size],))
    if device.type != "cuda":
        yield from chunks
        return
    transfer = torch.cuda.Stream(device=device)
    compute = torch.cuda.current_stream(device)

    def stage(chunk):
        pinned = chunk.pin_memory()
        with torch.cuda.stream(transfer):
            value = pinned.to(device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(transfer)
        return value, ready

    pending = None
    try:
        for chunk, is_last in chunks:
            following = (*stage(chunk), is_last)
            if pending is not None:
                value, ready, last = pending
                compute.wait_event(ready)
                value.record_stream(compute)
                yield value, last
            pending = following
        if pending is not None:
            value, ready, last = pending
            compute.wait_event(ready)
            value.record_stream(compute)
            yield value, last
    finally:
        transfer.synchronize()


@torch.no_grad()
def search(paths, maximum, params, device, *, candidate_batch_size=16,
           chunk_size=None, progress=None):
    if progress is not None:
        progress(0, 0)
    if maximum <= 0:
        if progress is not None:
            progress(params.search_num, 0)
        return 0.0, [0.0] * params.search_num
    device = torch.device(device)
    accumulator = CandidateSearch(maximum, params, device, candidate_batch_size, chunk_size)
    last_report = time.monotonic()
    for block, (h, is_last) in enumerate(_device_chunks(paths, accumulator.chunk_size, device), 1):
        accumulator._block(h, progress if is_last else None, block)
        if not is_last and progress is not None and time.monotonic() - last_report >= 5:
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
            progress(0, block)
            last_report = time.monotonic()
    return accumulator.result()


class CandidateSearch:
    """Bounded, device-resident SSE accumulation across activation batches."""

    def __init__(self, maximum, params, device, candidate_batch_size=16, chunk_size=None):
        self.params = params
        device = torch.device(device)
        candidates = torch.linspace(maximum * self.params.search_start, maximum,
                                    self.params.search_num, dtype=torch.float64).tolist()
        group_size = min(candidate_batch_size, len(candidates))
        # CUDA uses larger blocks to amortize per-step dispatch. With 16 candidates
        # and four thresholds this gives 1,048,576 activations. Keep this budget
        # for the single-step fallback too; temporal MTH fusion uses less memory.
        # Keep the conservative CPU budget and scale both
        # budgets down with candidate/threshold counts for the eager fallback.
        # Time does not enter the bound because each spike is decoded immediately.
        if chunk_size is None:
            workspace_elements = 536_870_912 if device.type == "cuda" else 4_194_304
            chunk_size = max(1, workspace_elements //
                             (group_size * max(8, 8 * self.params.num_thresholds)))
        errors = torch.zeros(len(candidates), dtype=torch.float64, device=device)
        groups = []
        for start in range(0, len(candidates), group_size):
            values = candidates[start:start + group_size]
            # Python scalar thresholds in the original search use the default dtype.
            threshold = torch.tensor(values, device=device).reshape(-1, 1)
            neuron = (IF(threshold) if self.params.neuron == "IF"
                      else MTH(threshold, self.params.num_thresholds)).eval()
            groups.append((start, len(values), neuron, threshold == 0))
        self.candidates = candidates
        self.groups = groups
        self.errors = errors
        self.chunk_size = chunk_size
        self.count = 0
        self.blocks = 0
        self.inactive = maximum <= 0

    @torch.no_grad()
    def update(self, activation):
        if self.inactive:
            self.count += activation.numel()
            return
        flat = activation.detach().reshape(-1)
        for h in flat.split(self.chunk_size):
            self._block(h)
            self.blocks += 1

    def _block(self, h, progress=None, block=0):
        params = self.params
        groups = self.groups
        errors = self.errors
        reference = h.relu().double()
        self.count += h.numel()
        for start, size, neuron, zero in groups:
            if params.neuron == "MTH":
                output = mth_const_in(h, neuron.threshold, params.time_steps,
                                      params.num_thresholds, params.coding)
            else:
                neuron.reset_state()
                inputs = h.unsqueeze(0).expand(size, -1)
                if inputs.is_cuda:
                    # Reuse the materialized candidate input over all steps.
                    inputs = inputs.contiguous()
                accumulated = None
                # Accumulate low-precision outputs in float32, like torch's
                # reduction kernels, and cast once at the decoder boundary.
                output_dtype = None
                for t in range(1, params.time_steps + 1):
                    spike = neuron(inputs).masked_fill(zero, 0)
                    output_dtype = spike.dtype
                    term = spike / t if params.coding == "differential_coding" else spike
                    if term.dtype in (torch.float16, torch.bfloat16):
                        term = term.float()
                    accumulated = term if accumulated is None else accumulated + term
                if params.coding == "rate_coding":
                    accumulated = accumulated / params.time_steps
                output = accumulated.to(output_dtype)
            errors[start:start + size] += (output.double() - reference).square().sum(dim=1)
            neuron.reset_state()
            if progress is not None:
                if not torch.isfinite(errors[start:start + size]).all().item():
                    raise ValueError("non-finite MSE during threshold search")
                progress(start + size, block)

    def result(self):
        if not self.count:
            raise ValueError("no calibration activations available for MMSE search")
        if self.inactive:
            return 0.0, [0.0] * self.params.search_num
        mses = (self.errors / self.count).cpu().tolist()
        if not all(math.isfinite(value) for value in mses):
            raise ValueError("non-finite MSE during threshold search")
        best = min(range(len(mses)), key=mses.__getitem__)
        return self.candidates[best], mses
