import types

import torch

from utils import MergeTemporalDim, ExpandTemporalDim


def add_dimention(x, T):
    x = x.unsqueeze(0)
    x = x.repeat(T, 1, 1, 1, 1)
    return x


def _run_backbone_steps(model, x):
    T = model.T
    x = add_dimention(x, T)
    x = model.merge(x)
    out = model.init_forward_backbone(x)
    return model.expand(out)


def _squeeze_attnpool_out(out):
    if out.dim() == 3 and out.shape[0] == 1:
        return out.squeeze(0)
    return out


def _prefix_rate_coding(ap_out):
    cumsum = torch.cumsum(ap_out, dim=0)
    idx = torch.arange(1, ap_out.shape[0] + 1, device=ap_out.device, dtype=ap_out.dtype)
    return cumsum / idx.view(-1, 1, 1)


def _reset_module_tree(module):
    for name, child in module._modules.items():
        if hasattr(child, "_modules"):
            _reset_module_tree(child)
        if hasattr(child, "reset"):
            child.reset()


def _forward_attnpool_steps(model, backbone_out):
    _reset_module_tree(model.attnpool)
    ap_outs = []
    for t in range(model.T):
        ap_outs.append(_squeeze_attnpool_out(model.forward_attnpool(backbone_out[t])))
    return torch.stack(ap_outs, dim=0)


def encode_backbone_aggregate(model, x):
    if model.coding_type != "rate" or model.step_mode != "m":
        raise ValueError(
            f"encode_backbone_aggregate only supports rate+step_mode=m, "
            f"got {model.coding_type}+{model.step_mode}"
        )
    back_out = _run_backbone_steps(model, x)
    return back_out.mean(dim=0)


def forward_clip_snn_rate_m(self, x):
    back_out = _run_backbone_steps(self, x)
    outputs = []
    for t in range(self.T):
        feat = back_out[: t + 1].mean(dim=0)
        outputs.append(_squeeze_attnpool_out(self.forward_attnpool(feat)))
    return torch.stack(outputs, dim=0)


def forward_clip_snn_rate_m_full(self, x):
    back_out = _run_backbone_steps(self, x)
    ap_out = _forward_attnpool_steps(self, back_out)
    return _prefix_rate_coding(ap_out)


def forward_replace_clip(args, model):
    if args.coding_type != "rate":
        raise ValueError(
            f"CLIP SNN forward only supports coding_type=rate, got {args.coding_type}"
        )
    if args.step_mode != "m":
        raise ValueError(
            f"CLIP SNN forward only supports step_mode=m, got {args.step_mode}"
        )

    model.coding_type = args.coding_type
    model.step_mode = args.step_mode
    model.convert_attn = getattr(args, "convert_attn", False)
    model.init_forward_backbone = model.forward_backbone
    model.T = args.time
    model.merge = MergeTemporalDim()
    model.expand = ExpandTemporalDim(model.T)

    if model.convert_attn:
        model.init_forward = model.forward
        model.forward = types.MethodType(forward_clip_snn_rate_m_full, model)
    else:
        model.forward = types.MethodType(forward_clip_snn_rate_m, model)

    return model
