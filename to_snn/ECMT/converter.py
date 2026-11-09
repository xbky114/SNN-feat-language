"""ECMT conversion for transformer visual towers.

The converter owns plan validation, streaming threshold calibration, model copying,
and artifact construction.  Stateful ECMT operators are supplied by ``to_snn.ECMT``
state modules and are deliberately resolved lazily so this module remains usable
with CPU-only installations.
"""
from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
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


def _input_spec(model, fallback_device):
    """Return the device and dtype expected at a model's tensor input.

    OpenAI CLIP keeps some transformer parameters in fp32 while converting
    convolution/linear weights to fp16.  Looking at an arbitrary first
    parameter therefore is not sufficient to prepare image batches: the
    dtype must match the first operator that consumes the image.  ViT models
    expose that operator as ``conv1``; other models fall back to their first
    floating point parameter.
    """
    input_module = getattr(model, "conv1", None)
    input_weight = getattr(input_module, "weight", None)
    if isinstance(input_weight, torch.Tensor):
        return input_weight.device, input_weight.dtype

    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.device, parameter.dtype
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return buffer.device, buffer.dtype
    return fallback_device, None


def _cast_input_to_visual_dtype(module, args, kwargs):
    """Adapt an ECMT ViT's public input to its patch-convolution dtype."""
    if not args or not isinstance(args[0], torch.Tensor):
        return args, kwargs
    input_weight = getattr(getattr(module, "conv1", None), "weight", None)
    value = args[0]
    if isinstance(input_weight, torch.Tensor) and value.is_floating_point():
        value = value.to(device=input_weight.device, dtype=input_weight.dtype)
        args = (value, *args[1:])
    return args, kwargs


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
        self._sample_count = 0

    def add(self, x):
        values = x.detach().reshape(-1)
        total = values.numel()
        if not total:
            return
        self.count += total
        # A ViT-L/14 seam can contain tens of millions of values per batch.
        # Select a bounded, evenly spread candidate set on-device before the
        # CPU transfer; moving the full activation made calibration dominated
        # by synchronization and host-side sampling.
        if total > self.limit:
            indices = torch.arange(self.limit, device=values.device, dtype=torch.long)
            indices = indices * (total - 1) // (self.limit - 1)
            values = values.index_select(0, indices)
        values = values.float().cpu()
        previous = self._sample_count
        self._sample_count += values.numel()
        filled = min(self.limit - self.values.numel(), values.numel())
        if filled:
            self.values = torch.cat((self.values, values[:filled]))
            values = values[filled:]
        if not values.numel():
            return
        # Deterministic reservoir sampling keeps calibration memory bounded.
        # Every item in the stream has the same chance of occupying a slot.
        positions = torch.arange(previous + filled, self._sample_count, dtype=torch.int64)
        slots = torch.remainder(positions * 1103515245 + 12345, positions + 1)
        keep = slots < self.limit
        if keep.any():
            self.values[slots[keep]] = values[keep]

    def quantile(self, q):
        if not self.values.numel():
            raise ValueError("empty calibration activation")
        return float(torch.quantile(self.values, q).item())


def _calibrate(model, plan, loader, device, percent):
    if loader is None:
        raise ValueError("calibration_loader is required for ECMT")
    model_device, model_dtype = _input_spec(model, device)
    neuron_targets = [
        target for target in plan.targets
        if str(target.kind).lower() in _NEURON_KINDS
        and str(target.kind).lower() != "softmax_neuron"
    ]
    print(
        f"[ECMT] calibrating {len(neuron_targets)} neuron seams "
        f"(percent={percent:g})",
        flush=True,
    )
    stats = {t.path: _Reservoir() for t in neuron_targets}
    finite = {
        target.path: torch.ones((), dtype=torch.bool, device=model_device)
        for target in neuron_targets
    }
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None

    def run_pass():
        hooks = []
        for target in neuron_targets:
            def capture(_module, _inputs, output, name=target.path):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"ECMT calibration target {name!r} must return a tensor")
                if output.numel() == 0:
                    raise ValueError(f"non-finite or empty calibration activation at {name!r}")
                finite[name] = finite[name] & torch.isfinite(output).all()
                stats[name].add(output)
            hooks.append(model.get_submodule(target.path).register_forward_hook(capture))
        samples = 0
        batches = 0
        try:
            with torch.no_grad():
                for batches, batch in enumerate(loader, 1):
                    if batches == 1:
                        total = total_batches if total_batches is not None else "?"
                        print(
                            f"[ECMT] calibration batch {batches}/{total} started",
                            flush=True,
                        )
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
                    if batches % 10 == 0 or batches == total_batches:
                        total = total_batches if total_batches is not None else "?"
                        print(
                            f"[ECMT] calibration batch {batches}/{total} complete "
                            f"(samples={samples})",
                            flush=True,
                        )
        finally:
            for hook in hooks:
                hook.remove()
        return samples, batches

    samples, batches = run_pass()
    if samples == 0:
        raise ValueError("calibration loader is empty")
    print(
        f"[ECMT] calibration complete (batches={batches}, samples={samples})",
        flush=True,
    )
    if not torch.stack(tuple(finite.values())).all().item():
        raise ValueError("non-finite calibration activation")
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
    print(f"[ECMT] conversion started (targets={len(plan.targets)})", flush=True)
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
    # SNNArtifact intentionally keeps its generic parameter-based boundary
    # adaptation for every model family.  OpenAI ViT has fp32 embeddings
    # registered before its fp16 input convolution, so attach this
    # model-specific adaptation to the ECMT artifact instead of changing the
    # shared artifact protocol.
    if isinstance(getattr(model, "conv1", None), nn.Conv2d):
        model.register_forward_pre_hook(_cast_input_to_visual_dtype, with_kwargs=True)
    artifact = SNNArtifact(model, params, plan).eval()
    artifact.reset_state()
    print(f"[ECMT] conversion finished (targets={len(plan.targets)})", flush=True)
    return artifact, info
