"""Bounded candidate batching and disk prefetch for independent MMSE searches."""
from concurrent.futures import ThreadPoolExecutor
import time

import torch

from .neurons import IFRate, MTHRate


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
    candidates = torch.linspace(maximum * params.search_start, maximum,
                                params.search_num, dtype=torch.float64).tolist()
    group_size = min(candidate_batch_size, len(candidates))
    # Bound the expanded candidate x threshold-level intermediates. Time does
    # not enter this bound because decoding consumes each spike immediately.
    if chunk_size is None:
        chunk_size = max(1, 4_194_304 // (group_size * max(8, 8 * params.num_thresholds)))
    errors = torch.zeros(len(candidates), dtype=torch.float64, device=device)
    groups = []
    for start in range(0, len(candidates), group_size):
        values = candidates[start:start + group_size]
        # Python scalar thresholds in the original search use the default dtype.
        threshold = torch.tensor(values, device=device).reshape(-1, 1)
        neuron = (IFRate(threshold) if params.neuron == "IF"
                  else MTHRate(threshold, params.num_thresholds)).eval()
        groups.append((start, len(values), neuron, threshold == 0))
    count = 0
    last_report = time.monotonic()
    for block, (h, is_last) in enumerate(_device_chunks(paths, chunk_size, device), 1):
        reference = h.relu().double()
        count += h.numel()
        for start, size, neuron, zero in groups:
            neuron.reset_state()
            inputs = h.unsqueeze(0).expand(size, -1)
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
            if is_last and progress is not None:
                # Validate and synchronize the completed group before reporting.
                if not torch.isfinite(errors[start:start + size]).all().item():
                    raise ValueError("non-finite MSE during threshold search")
                progress(start + size, block)
        if not is_last and progress is not None and time.monotonic() - last_report >= 5:
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
            progress(0, block)
            last_report = time.monotonic()
    if not count:
        raise ValueError("no calibration activations available for MMSE search")
    mses = (errors / count).cpu().tolist()
    if not all(torch.isfinite(torch.tensor(mses, dtype=torch.float64))):
        raise ValueError("non-finite MSE during threshold search")
    best = min(range(len(mses)), key=mses.__getitem__)
    return candidates[best], mses
