"""Calibrate independent ReLU replacements with two streaming ANN passes."""

import copy
from collections.abc import Iterator

import torch
from torch import nn

from to_snn.prototype import SNNArtifact
from to_snn.validation import reject_unsupported_modules
from to_snn.neurons import IF, MTH
from .config import MMSEConversionParams
from .search import CandidateSearch


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


def _run_pass(model, plan, loader, device, capture, stage):
    hooks = []
    samples = 0
    try:
        for target in plan.targets:
            hooks.append(model.get_submodule(target.path).register_forward_pre_hook(
                lambda module, inputs, name=target.path: capture(name, inputs[0].detach())))
        try:
            total = len(loader)
        except TypeError:
            total = "?"
        for index, batch in enumerate(loader, 1):
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            if not isinstance(images, torch.Tensor):
                raise TypeError("calibration batches must be tensors or (images, labels)")
            model(images.to(device, non_blocking=True))
            samples += images.shape[0]
            if index == 1 or index % 10 == 0 or index == total:
                if device.type == "cuda":
                    torch.cuda.current_stream(device).synchronize()
                print(f"[MMSE] pass={stage}/2 batches={index}/{total} samples={samples}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    return samples


@torch.no_grad()
def convert(visual, plan, params, calibration_loader=None, *, layer_workers=1,
            candidate_batch_size=16, chunk_size=None):
    """Two ANN passes: maxima, then streaming SSE; no activation cache.

    The loader must be reiterable with identical samples and deterministic
    preprocessing on both passes. Hooks run on the ANN's current CUDA stream,
    so inplace ReLUs cannot overwrite inputs before search consumes them.
    layer_workers is retained as a validated caller option; layer execution
    follows the ANN forward order without thread pools or activation copies.
    """
    if not isinstance(params, MMSEConversionParams):
        raise TypeError("params must be MMSEConversionParams")
    params.__post_init__()
    for name, value in (("layer_workers", layer_workers),
                        ("candidate_batch_size", candidate_batch_size), ("chunk_size", chunk_size)):
        if (name != "chunk_size" or value is not None) and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    _validate_plan(visual, plan)
    if calibration_loader is None:
        raise ValueError("calibration_loader is required for MMSE")
    if isinstance(calibration_loader, Iterator):
        raise ValueError("two-pass MMSE requires a reiterable calibration loader, not a one-shot iterator")
    model = copy.deepcopy(visual).eval()
    state = next(iter(model.parameters()), None)
    if state is None:
        state = next(iter(model.buffers()), torch.empty(0))
    device = state.device
    maxima = {}
    finite = {}
    counts = {target.path: 0 for target in plan.targets}

    def maximum(name, activation):
        if not activation.numel():
            raise ValueError(f"empty preactivation at {name!r}")
        value = activation.amax()
        valid = torch.isfinite(activation).all()
        maxima[name] = torch.maximum(maxima[name], value) if name in maxima else value
        finite[name] = finite[name] & valid if name in finite else valid
        counts[name] += activation.numel()

    print(f"[MMSE] two-pass streaming calibration: {len(plan.targets)} ReLU layers", flush=True)
    samples = _run_pass(model, plan, calibration_loader, device, maximum, 1)
    if not samples or len(maxima) != len(plan.targets):
        raise ValueError("no calibration activations recorded for one or more layers")
    names = [target.path for target in plan.targets]
    if not torch.stack([finite[name] for name in names]).all().item():
        raise ValueError("non-finite calibration preactivations")
    values = torch.stack([maxima[name] for name in names]).cpu().tolist()
    maxima = dict(zip(names, values))
    searches = {name: CandidateSearch(maxima[name], params, device,
                                     candidate_batch_size, chunk_size) for name in names}
    second_finite = torch.ones((), dtype=torch.bool, device=device)

    def accumulate(name, activation):
        nonlocal second_finite
        second_finite = second_finite & torch.isfinite(activation).all()
        searches[name].update(activation)

    second_samples = _run_pass(model, plan, calibration_loader, device, accumulate, 2)
    if samples != second_samples or any(searches[name].count != counts[name] for name in names):
        raise ValueError("calibration loader produced different activation counts across passes")
    if not second_finite.item():
        raise ValueError("non-finite calibration preactivations in second pass")
    # Select every threshold before mutating the isolated ANN copy.
    results = {name: searches[name].result() for name in names}
    info = {}
    for index, name in enumerate(names, 1):
        threshold, mses = results[name]
        parent, _, child = name.rpartition(".")
        holder = model.get_submodule(parent) if parent else model
        setattr(holder, child, _neuron(threshold, params).to(device).eval())
        info[name] = {"max_pre_act": maxima[name], "min_MSE": min(mses), "MSEs": mses}
        print(f"[MMSE] {index}/{len(names)} {name}: threshold={threshold:.8g}, MSE={min(mses):.8g}", flush=True)
    artifact = SNNArtifact(model, params, plan).eval()
    artifact.reset_state()
    return artifact, info
