"""ECMT conversion for transformer visual towers.

The converter owns plan validation, streaming threshold calibration, model copying,
and artifact construction.  Stateful ECMT operators are supplied by ``to_snn.ECMT``
state modules and are deliberately resolved lazily so this module remains usable
with CPU-only installations.
"""
from __future__ import annotations

import copy
import importlib
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import nn

from to_snn.prototype import SNNArtifact
from .config import ECMTConversionParams

_KINDS = {
    "neuron", "linear", "qkv", "q", "k", "v", "softmax_neuron",
    "layernorm", "ln", "quickgelu", "gelu", "softmax", "qk", "av",
}
_NEURON_KINDS = {"neuron", "linear", "qkv", "q", "k", "v", "softmax_neuron"}
_REQUIRED_KINDS = {
    "layernorm", "qkv", "q", "k", "v", "qk", "softmax",
    "softmax_neuron", "av", "linear", "quickgelu",
}
_VIT_BLOCK_SEAMS = (
    ("ln_1", "layernorm"),
    ("attn.qkv_neuron", "qkv"),
    ("attn.q_neuron", "q"),
    ("attn.k_neuron", "k"),
    ("attn.v_neuron", "v"),
    ("attn.qk", "qk"),
    ("attn.softmax", "softmax"),
    ("attn.softmax_neuron", "softmax_neuron"),
    ("attn.av", "av"),
    ("attn.x_out_neuron", "linear"),
    ("mlp.c_fc_neuron", "linear"),
    ("mlp.gelu", "quickgelu"),
    ("mlp.c_proj_neuron", "linear"),
    ("ln_2", "layernorm"),
)


class ZeroOutput(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def _images(batch):
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, Mapping):
        value = batch.get("image")
        if value is None:
            raise TypeError("calibration mapping must contain an 'image' tensor")
        return value
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError("calibration batches must be tensors, (images, labels), or mappings with image")


def _state_class(*names):
    for module_name in ("to_snn.ECMT.state", "to_snn.ECMT.modules", "to_snn.ECMT.compensation"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in names:
            cls = getattr(module, name, None)
            if cls is not None:
                return cls
    raise ImportError("ECMT state modules are unavailable; install to_snn.ECMT state implementation")


def _wrap_function(module, kind):
    cls = _state_class("ExpectationCompensation", "ECMTExpectation", "ExpectationCompensated")
    # The reference state implementation accepts a callable module.  Keep a
    # small compatibility fallback for positional/keyword constructor variants.
    try:
        return cls(module)
    except TypeError:
        try:
            return cls(function=module)
        except TypeError:
            return cls(op=module)


def _wrap_matmul(module, kind):
    cls = _state_class("ExpectationCompensationMatmul", "ECMTMatmul", "CompensatedMatmul")
    try:
        return cls(module)
    except TypeError:
        try:
            return cls(op=module)
        except TypeError:
            return cls()


def _neuron(pos, neg, levels, params):
    cls = getattr(importlib.import_module("to_snn.neurons"), "TwoSide", None)
    if cls is None:
        cls = getattr(importlib.import_module("to_snn.neurons"), "TwoSideNeuron")
    try:
        return cls(pos, neg, levels, gap=params.gap, fused=params.fused)
    except TypeError:
        return cls(positive_threshold=pos, negative_threshold=neg,
                   num_thresholds=levels, gap=params.gap, fused=params.fused)


def _validate_plan(model, plan):
    if plan is None or not getattr(plan, "targets", None):
        raise ValueError("ECMT requires a nonempty conversion plan")
    paths = [t.path for t in plan.targets]
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("ECMT plan targets must have nonempty paths")
    if len(paths) != len(set(paths)):
        raise ValueError("ECMT plan contains duplicate target paths")
    for target in plan.targets:
        kind = str(target.kind).lower()
        if kind not in _KINDS:
            raise ValueError(f"unknown ECMT target kind: {target.kind!r}")
        try:
            module = model.get_submodule(target.path)
        except AttributeError as exc:
            raise ValueError(f"ECMT target path does not exist: {target.path!r}") from exc
        if not isinstance(module, nn.Module):
            raise ValueError(f"ECMT target is not a module: {target.path!r}")
    metadata = getattr(plan, "metadata", {}) or {}
    required = metadata.get("required_kinds")
    if not required:
        raise ValueError("ECMT plan must declare required_kinds metadata")
    required = {str(kind).lower() for kind in required}
    unknown_required = required - _KINDS
    if unknown_required:
        raise ValueError(f"ECMT plan declares unknown required kinds: {sorted(unknown_required)}")
    missing_categories = _REQUIRED_KINDS - required
    if missing_categories:
        raise ValueError(
            "ECMT plan is incomplete; metadata is missing target kinds: "
            f"{sorted(missing_categories)}"
        )
    present = {str(t.kind).lower() for t in plan.targets}
    missing = required - present
    if missing:
        raise ValueError(f"ECMT plan is incomplete; missing target kinds: {sorted(missing)}")
    expected_paths = metadata.get("expected_paths")
    if not expected_paths:
        raise ValueError("ECMT plan must declare expected_paths metadata")
    if any(not isinstance(path, str) or not path for path in expected_paths):
        raise ValueError("ECMT plan expected_paths must contain nonempty strings")
    if len(expected_paths) != len(set(expected_paths)):
        raise ValueError("ECMT plan expected_paths contains duplicates")
    expected_paths = set(expected_paths)
    actual_paths = set(paths)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise ValueError(
            f"ECMT plan paths do not match its declared coverage: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if metadata.get("backend") == "openai_vit_L_14":
        blocks = getattr(getattr(model, "transformer", None), "resblocks", None)
        if blocks is None:
            raise ValueError("OpenAI ViT-L/14 ECMT plan requires transformer.resblocks")
        model_paths = {"ln_pre"}
        for index in range(len(blocks)):
            prefix = f"transformer.resblocks.{index}"
            model_paths.update(f"{prefix}.{suffix}" for suffix, _kind in _VIT_BLOCK_SEAMS)
        if actual_paths != model_paths:
            missing = sorted(model_paths - actual_paths)
            unexpected = sorted(actual_paths - model_paths)
            raise ValueError(
                "OpenAI ViT-L/14 ECMT plan does not cover every transformer seam: "
                f"missing={missing}, unexpected={unexpected}"
            )


class _Reservoir:
    def __init__(self, limit=100_000):
        self.limit = limit
        self.values = torch.empty(0, dtype=torch.float32)
        self.count = 0

    def add(self, x):
        x = x.detach().float().reshape(-1).cpu()
        if not x.numel():
            return
        previous = self.count
        self.count += x.numel()
        filled = min(self.limit - self.values.numel(), x.numel())
        if filled:
            self.values = torch.cat((self.values, x[:filled]))
            x = x[filled:]
        if not x.numel():
            return
        # Deterministic reservoir sampling keeps calibration memory bounded.
        # Every item in the stream has the same chance of occupying a slot.
        start = previous + filled
        positions = torch.arange(start, self.count, dtype=torch.int64)
        slots = torch.remainder(positions * 1103515245 + 12345, positions + 1)
        keep = slots < self.limit
        if keep.any():
            self.values[slots[keep]] = x[keep]

    def quantile(self, q):
        if not self.values.numel():
            raise ValueError("empty calibration activation")
        return float(torch.quantile(self.values, q).item())


def _calibrate(model, plan, loader, device, percent):
    if loader is None:
        raise ValueError("calibration_loader is required for ECMT")
    if isinstance(loader, Iterator):
        raise ValueError("ECMT calibration requires a reiterable loader, not a one-shot iterator")
    model_state = next(iter(model.parameters()), None)
    model_state = model_state if model_state is not None else next(
        iter(model.buffers()), torch.empty(0, device=device)
    )
    model_device = model_state.device
    model_dtype = model_state.dtype if model_state.is_floating_point() else None
    neuron_targets = [
        target for target in plan.targets
        if str(target.kind).lower() in _NEURON_KINDS
        and str(target.kind).lower() != "softmax_neuron"
    ]
    stats = {t.path: _Reservoir() for t in neuron_targets}
    def run_pass(collect):
        hooks = []
        counts = {target.path: 0 for target in neuron_targets}
        for target in neuron_targets:
            def capture(_module, _inputs, output, name=target.path):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"ECMT calibration target {name!r} must return a tensor")
                if output.numel() == 0 or not torch.isfinite(output).all():
                    raise ValueError(f"non-finite or empty calibration activation at {name!r}")
                counts[name] += output.numel()
                if collect:
                    stats[name].add(output)
            hooks.append(model.get_submodule(target.path).register_forward_hook(capture))
        samples = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    images = _images(batch)
                    if not isinstance(images, torch.Tensor):
                        raise TypeError("calibration image must be a torch.Tensor")
                    if images.ndim == 0 or images.shape[0] == 0:
                        raise ValueError("empty calibration batch")
                    if not torch.isfinite(images).all():
                        raise ValueError("non-finite calibration input")
                    if model_dtype is None:
                        model_input = images.to(device=model_device)
                    else:
                        model_input = images.to(device=model_device, dtype=model_dtype)
                    model(model_input)
                    samples += int(images.shape[0])
        finally:
            for hook in hooks:
                hook.remove()
        return samples, counts

    samples, counts = run_pass(collect=True)
    if samples == 0:
        raise ValueError("calibration loader is empty")
    second_samples, second_counts = run_pass(collect=False)
    if samples != second_samples or counts != second_counts:
        raise ValueError("calibration loader changed between passes")
    result = {}
    for target in neuron_targets:
        stat = stats[target.path]
        if stat.count == 0:
            raise ValueError(f"no calibration activation recorded for {target.path!r}")
        pos = stat.quantile(percent)
        neg = -stat.quantile(1.0 - percent)
        # Quantiles over a one-sided activation are inactive on the other side.
        if not torch.isfinite(torch.tensor([pos, neg])).all():
            raise ValueError(f"non-finite calibration activation at {target.path!r}")
        if pos == 0 and neg == 0:
            result[target.path] = {"positive": 0.0, "negative": 0.0,
                                   "samples": samples, "elements": stat.count}
        elif pos <= 0 or neg <= 0:
            raise ValueError(f"calibration tails are inactive at {target.path!r}")
        else:
            result[target.path] = {"positive": pos, "negative": neg,
                                   "samples": samples, "elements": stat.count}
    return result


def _replace(model, plan, thresholds, params):
    for target in plan.targets:
        kind = str(target.kind).lower()
        parent, _, name = target.path.rpartition(".")
        holder = model.get_submodule(parent) if parent else model
        original = model.get_submodule(target.path)
        if kind in ("layernorm", "ln", "quickgelu", "gelu", "softmax"):
            replacement = _wrap_function(original, kind)
        elif kind in ("qk", "av"):
            replacement = _wrap_matmul(original, kind)
        elif kind in _NEURON_KINDS:
            info = thresholds.get(target.path, {"positive": 0.0, "negative": 0.0})
            if kind != "softmax_neuron" and info["positive"] == 0 and info["negative"] == 0:
                replacement = nn.Sequential(original, ZeroOutput())
            else:
                levels = params.softmax_num if kind == "softmax_neuron" else (params.qkv_num if kind in {"qkv", "q", "k", "v"} else params.linear_num)
                pos = params.softmax_p if kind == "softmax_neuron" else info["positive"]
                neg = params.softmax_p if kind == "softmax_neuron" else info["negative"]
                replacement = nn.Sequential(original, _neuron(pos, neg, levels, params))
        else:
            raise ValueError(f"unsupported ECMT target kind: {target.kind!r}")
        setattr(holder, name, replacement)


@torch.no_grad()
def convert(visual, plan, params, calibration_loader=None, **_kwargs):
    if not isinstance(params, ECMTConversionParams):
        raise TypeError("params must be ECMTConversionParams")
    params.__post_init__()
    _validate_plan(visual, plan)
    # Obtain device from the ANN before making any modifications.
    state = next(iter(visual.parameters()), None)
    state = state if state is not None else next(iter(visual.buffers()), torch.empty(0))
    device = state.device
    calibration_model = copy.deepcopy(visual).eval()
    thresholds = _calibrate(calibration_model, plan, calibration_loader, device, params.percent)
    for target in plan.targets:
        if str(target.kind).lower() == "softmax_neuron":
            thresholds[target.path] = {
                "positive": params.softmax_p,
                "negative": params.softmax_p,
                "samples": 0,
                "elements": 0,
            }
    model = copy.deepcopy(visual).eval()
    _replace(model, plan, thresholds, params)
    info = {"params": {k: getattr(params, k) for k in params.__dataclass_fields__},
            "thresholds": thresholds,
            "targets": [{"path": t.path, "kind": t.kind} for t in plan.targets]}
    artifact = SNNArtifact(model, params, plan).eval()
    artifact.reset_state()
    return artifact, info
