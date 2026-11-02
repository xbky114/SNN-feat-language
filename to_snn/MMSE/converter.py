"""Calibrate independent ReLU replacements against cached ANN preactivations."""

import copy
from concurrent.futures import ThreadPoolExecutor
import math
from threading import Lock
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from to_snn.prototype import SNNArtifact
from to_snn.validation import reject_unsupported_modules
from to_snn.neurons import IF, MTH
from .config import MMSEConversionParams
from .search import search as _search


class ZeroOutput(nn.Module):
    """A zero-threshold candidate, also used for inactive calibration layers."""

    def __init__(self):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(0.0))

    def forward(self, x):
        return torch.zeros_like(x)


def _neuron(threshold, params):
    if threshold == 0:
        return ZeroOutput()
    if params.neuron == "IF":
        return IF(threshold)
    return MTH(threshold, params.num_thresholds)


def _validate_plan(model, plan):
    reject_unsupported_modules(model, "MMSE")
    paths = [target.path for target in plan.targets]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("MMSE requires a nonempty plan with unique ReLU targets")
    for target in plan.targets:
        if (not target.path or target.kind != "relu"
                or not isinstance(model.get_submodule(target.path), nn.ReLU)):
            raise ValueError(f"MMSE target must expose a ReLU module: {target.path!r}")
    relus = {name for name, module in model.named_modules() if isinstance(module, nn.ReLU)}
    if set(paths) != relus:
        raise ValueError("MMSE conversion plan must expose every ReLU module")


def _cache_preactivations(model, plan, loader, directory, device):
    maxima = {target.path: -math.inf for target in plan.targets}
    files = {target.path: [] for target in plan.targets}
    hooks = []

    def capture(name, folder, inputs):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError(f"ReLU input at {name!r} must be a tensor")
        # Copy before an inplace ReLU can overwrite the input. Keep no graph
        # or full-dataset activation tensor in host/device memory.
        activation = inputs[0].detach().to(device="cpu", copy=True)
        if not activation.numel() or not torch.isfinite(activation).all():
            raise ValueError(f"empty or non-finite preactivation at {name!r}")
        maxima[name] = max(maxima[name], activation.max().item())
        path = folder / f"{len(files[name])}.pt"
        torch.save(activation, path)
        files[name].append(path)

    try:
        for index, target in enumerate(plan.targets):
            folder = directory / str(index)
            folder.mkdir()
            hooks.append(model.get_submodule(target.path).register_forward_pre_hook(
                lambda module, inputs, name=target.path, folder=folder:
                    capture(name, folder, inputs)))
        model.eval()
        for index, batch in enumerate(loader, 1):
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            if not isinstance(images, torch.Tensor):
                raise TypeError("calibration batches must be tensors or (images, labels)")
            model(images.to(device))
            if index == 1 or index % 10 == 0:
                print(f"[MMSE] cached calibration batch {index}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    missing = [name for name, paths in files.items() if not paths]
    if missing:
        raise ValueError(f"no calibration activations recorded for: {', '.join(missing)}")
    return maxima, files


@torch.no_grad()
def convert(visual, plan, params, calibration_loader=None, *, layer_workers=4,
            candidate_batch_size=16, chunk_size=None):
    """Return ``(SNNArtifact, JSON-serializable search info)``.

    Calibration runs once, in eval mode, on an isolated ANN copy. Temporary
    activation files are removed on both success and failure. Only planned
    module-based ReLUs are supported; the plan must expose all such layers.
    layer_workers bounds concurrent searches; candidate_batch_size batches
    independent thresholds. chunk_size optionally overrides the activation
    block size. These controls do not change calibration/search parameters.
    """
    if not isinstance(params, MMSEConversionParams):
        raise TypeError("params must be MMSEConversionParams")
    params.__post_init__()
    for name, value in (("layer_workers", layer_workers),
                        ("candidate_batch_size", candidate_batch_size),
                        ("chunk_size", chunk_size)):
        if (name != "chunk_size" or value is not None) and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    _validate_plan(visual, plan)
    if calibration_loader is None:
        raise ValueError("calibration_loader is required for MMSE")
    model = copy.deepcopy(visual).eval()
    state = next(iter(model.parameters()), None)
    if state is None:
        state = next(iter(model.buffers()), torch.empty(0))
    device = state.device
    info = {}
    print(f"[MMSE] calibrating {len(plan.targets)} ReLU layers in eval mode", flush=True)
    with TemporaryDirectory(prefix="snn_mmse_") as temporary:
        maxima, files = _cache_preactivations(
            model, plan, calibration_loader, Path(temporary), device)
        progress_lock = Lock()
        finished_layers = set()

        def report_progress(name, candidates, blocks):
            with progress_lock:
                if candidates == params.search_num:
                    finished_layers.add(name)
                print(f"[MMSE] layers completed={len(finished_layers)}/{len(plan.targets)} | "
                      f"layer={name} | candidates completed={candidates}/{params.search_num} | "
                      f"activation blocks processed={blocks}", flush=True)

        def search_layer(target):
            def run():
                return _search(files[target.path], maxima[target.path], params, device,
                               candidate_batch_size=candidate_batch_size,
                               chunk_size=chunk_size,
                               progress=lambda candidates, blocks:
                                   report_progress(target.path, candidates, blocks))
            if device.type == "cuda":
                with torch.cuda.device(device):
                    stream = torch.cuda.Stream(device=device)
                    with torch.cuda.stream(stream):
                        result = run()
                    stream.synchronize()
                    return result
            return run()

        # Bounded submission; model mutations and reporting stay on caller.
        with ThreadPoolExecutor(max_workers=layer_workers) as pool:
            targets = iter(plan.targets)
            pending = []
            for _ in range(min(layer_workers, len(plan.targets))):
                target = next(targets)
                pending.append((target, pool.submit(search_layer, target)))
            index = 0
            while pending:
                target, future = pending.pop(0)
                threshold, mses = future.result()
                index += 1
                parent, _, name = target.path.rpartition(".")
                holder = model.get_submodule(parent) if parent else model
                setattr(holder, name, _neuron(threshold, params).to(device).eval())
                maximum = maxima[target.path]
                info[target.path] = {"max_pre_act": maximum,
                                     "min_MSE": min(mses), "MSEs": mses}
                for path in files[target.path]:
                    path.unlink()
                print(f"[MMSE] {index}/{len(plan.targets)} {target.path}: "
                      f"threshold={threshold:.8g}, MSE={min(mses):.8g}", flush=True)
                target = next(targets, None)
                if target is not None:
                    pending.append((target, pool.submit(search_layer, target)))
    artifact = SNNArtifact(model, params, plan).eval()
    artifact.reset_state()
    return artifact, info
