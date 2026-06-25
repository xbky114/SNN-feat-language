import types

import torch

from utils import MergeTemporalDim, ExpandTemporalDim


def add_dimention(x, T):
    x = x.unsqueeze(0)
    x = x.repeat(T, 1, 1, 1, 1)
    return x


def add_dimention_diff(x, T):
    x = x.unsqueeze(0)
    x = x.repeat(T + 1, 1, 1, 1, 1)
    x = x.clone()
    x[0] = 0
    x[2:] = 0
    return x


def decodeoutput(x):
    out = torch.zeros_like(x[1:])
    T = x.shape[0] - 1
    exp_in = x[0].clone().detach()
    for t in range(T):
        out[t] = exp_in + x[t + 1] - x[0]
        exp_in = exp_in + (x[t + 1] - x[0]) / (t + 1)
    return out


def _aggregate_backbone_output(backbone_out):
    if backbone_out.dim() == 5:
        return backbone_out.mean(dim=0)
    return backbone_out


def encode_backbone_aggregate(model, x):
    if model.coding_type == 'rate':
        if model.step_mode == 's':
            outputs = []
            for _ in range(model.T):
                outputs.append(model.init_forward_backbone(x))
            return torch.stack(outputs, dim=0).mean(dim=0)
        x = add_dimention(x, model.T)
        x = model.merge(x)
        backbone_out = model.init_forward_backbone(x)
        backbone_out = model.expand(backbone_out)
        return _aggregate_backbone_output(backbone_out)
    elif model.coding_type == 'leaky_rate':
        if model.step_mode == 's':
            outputs = []
            mul = 1
            for _ in range(model.T):
                outputs.append(model.init_forward_backbone(x / mul))
                mul /= model.tau
            return torch.stack(outputs, dim=0).mean(dim=0)
        x = add_dimention(x, model.T)
        for i in range(1, model.T):
            x[i] = x[i - 1] / model.tau
        x = model.merge(x)
        backbone_out = model.init_forward_backbone(x)
        backbone_out = model.expand(backbone_out)
        mul = 1
        for i in range(model.T):
            backbone_out[i] *= mul
            mul /= model.tau
        return _aggregate_backbone_output(backbone_out)
    elif model.coding_type == 'diff_rate':
        if model.step_mode == 's':
            outputs = []
            outputs.append(model.init_forward_backbone(torch.zeros_like(x)))
            outputs.append(model.init_forward_backbone(x))
            for _ in range(model.T - 1):
                outputs.append(model.init_forward_backbone(torch.zeros_like(x)))
            return decodeoutput(torch.stack(outputs, dim=0)).mean(dim=0)
        x = add_dimention_diff(x, model.T)
        x = model.merge(x)
        backbone_out = model.init_forward_backbone(x)
        backbone_out = model.expand(backbone_out)
        backbone_out = decodeoutput(backbone_out)
        return _aggregate_backbone_output(backbone_out)
    elif model.coding_type == 'diff_leaky_rate':
        if model.step_mode == 's':
            outputs = []
            outputs.append(model.init_forward_backbone(torch.zeros_like(x)))
            outputs.append(model.init_forward_backbone(x))
            mul = 1
            for _ in range(1, model.T):
                mul /= model.tau
                outputs.append(model.init_forward_backbone(torch.zeros_like(x)))
            return decodeoutput(torch.stack(outputs, dim=0)).mean(dim=0)
        x = add_dimention_diff(x, model.T)
        x = model.merge(x)
        backbone_out = model.init_forward_backbone(x)
        backbone_out = model.expand(backbone_out)
        mul = 1
        for i in range(2, model.T + 1):
            mul /= model.tau
            backbone_out[i] *= mul
        backbone_out = decodeoutput(backbone_out)
        return _aggregate_backbone_output(backbone_out)
    raise ValueError(f"Unexpected coding_type: {model.coding_type}")


def _reset_module_tree(module):
    for name, child in module._modules.items():
        if hasattr(child, "_modules"):
            _reset_module_tree(child)
        if hasattr(child, "reset"):
            child.reset()


def _forward_attnpool_over_time(model, backbone_out):
    _reset_module_tree(model.attnpool)
    outputs = []
    for t in range(model.T):
        outputs.append(model.forward_attnpool(backbone_out[t]))
    return torch.stack(outputs, dim=0).mean(dim=0)


def _aggregate_visual_output(out):
    if out.dim() == 3:
        return out.mean(dim=0)
    return out


def forward_clip_snn_rate_m_full(self, x):
    x = add_dimention(x, self.T)
    x = self.merge(x)
    backbone_out = self.init_forward_backbone(x)
    backbone_out = self.expand(backbone_out)
    return _forward_attnpool_over_time(self, backbone_out)


def forward_clip_snn_diff_rate_m_full(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    backbone_out = self.init_forward_backbone(x)
    backbone_out = self.expand(backbone_out)
    backbone_out = decodeoutput(backbone_out)
    return _forward_attnpool_over_time(self, backbone_out)


def forward_clip_snn_leaky_rate_m_full(self, x):
    x = add_dimention(x, self.T)
    for i in range(1, self.T):
        x[i] = x[i - 1] / self.tau
    x = self.merge(x)
    backbone_out = self.init_forward_backbone(x)
    backbone_out = self.expand(backbone_out)
    mul = 1
    for i in range(self.T):
        backbone_out[i] *= mul
        mul /= self.tau
    return _forward_attnpool_over_time(self, backbone_out)


def forward_clip_snn_diff_leaky_rate_m_full(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    backbone_out = self.init_forward_backbone(x)
    backbone_out = self.expand(backbone_out)
    mul = 1
    for i in range(2, self.T + 1):
        mul /= self.tau
        backbone_out[i] *= mul
    backbone_out = decodeoutput(backbone_out)
    return _forward_attnpool_over_time(self, backbone_out)


def forward_clip_snn_rate_s_full(self, x):
    outputs = []
    for _ in range(self.T):
        outputs.append(self.init_forward(x))
    return torch.stack(outputs, dim=0).mean(dim=0)


def forward_clip_snn_diff_rate_s_full(self, x):
    outputs = []
    outputs.append(self.init_forward(torch.zeros_like(x)))
    outputs.append(self.init_forward(x))
    for _ in range(self.T - 1):
        outputs.append(self.init_forward(torch.zeros_like(x)))
    return decodeoutput(torch.stack(outputs, dim=0)).mean(dim=0)


def forward_clip_snn_leaky_rate_s_full(self, x):
    outputs = []
    mul = 1
    for _ in range(self.T):
        outputs.append(self.init_forward(x / mul))
        mul /= self.tau
    return torch.stack(outputs, dim=0).mean(dim=0)


def forward_clip_snn_diff_leaky_rate_s_full(self, x):
    outputs = []
    outputs.append(self.init_forward(torch.zeros_like(x)))
    outputs.append(self.init_forward(x))
    mul = 1
    for _ in range(1, self.T):
        mul /= self.tau
        outputs.append(self.init_forward(torch.zeros_like(x)))
    return decodeoutput(torch.stack(outputs, dim=0)).mean(dim=0)


def forward_clip_snn_rate_m(self, x):
    x = add_dimention(x, self.T)
    x = self.merge(x)
    backbone_out = self.init_forward_backbone(x)
    backbone_out = self.expand(backbone_out)
    aggregated = _aggregate_backbone_output(backbone_out)
    return self.forward_attnpool(aggregated)


def forward_clip_snn_diff_rate_m(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    backbone_out = self.init_forward_backbone(x)
    backbone_out = self.expand(backbone_out)
    backbone_out = decodeoutput(backbone_out)
    aggregated = _aggregate_backbone_output(backbone_out)
    return self.forward_attnpool(aggregated)


def forward_clip_snn_leaky_rate_m(self, x):
    x = add_dimention(x, self.T)
    for i in range(1, self.T):
        x[i] = x[i - 1] / self.tau
    x = self.merge(x)
    backbone_out = self.init_forward_backbone(x)
    backbone_out = self.expand(backbone_out)
    mul = 1
    for i in range(self.T):
        backbone_out[i] *= mul
        mul /= self.tau
    aggregated = _aggregate_backbone_output(backbone_out)
    return self.forward_attnpool(aggregated)


def forward_clip_snn_diff_leaky_rate_m(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    backbone_out = self.init_forward_backbone(x)
    backbone_out = self.expand(backbone_out)
    mul = 1
    for i in range(2, self.T + 1):
        mul /= self.tau
        backbone_out[i] *= mul
    backbone_out = decodeoutput(backbone_out)
    aggregated = _aggregate_backbone_output(backbone_out)
    return self.forward_attnpool(aggregated)


def forward_clip_snn_rate_s(self, x):
    backbone_outputs = []
    for _ in range(self.T):
        backbone_outputs.append(self.init_forward_backbone(x))
    aggregated = torch.stack(backbone_outputs, dim=0).mean(dim=0)
    return self.forward_attnpool(aggregated)


def forward_clip_snn_diff_rate_s(self, x):
    backbone_outputs = []
    backbone_outputs.append(self.init_forward_backbone(torch.zeros_like(x)))
    backbone_outputs.append(self.init_forward_backbone(x))
    for _ in range(self.T - 1):
        backbone_outputs.append(self.init_forward_backbone(torch.zeros_like(x)))
    aggregated = decodeoutput(torch.stack(backbone_outputs, dim=0)).mean(dim=0)
    return self.forward_attnpool(aggregated)


def forward_clip_snn_leaky_rate_s(self, x):
    backbone_outputs = []
    mul = 1
    for _ in range(self.T):
        backbone_outputs.append(self.init_forward_backbone(x / mul))
        mul /= self.tau
    aggregated = torch.stack(backbone_outputs, dim=0).mean(dim=0)
    return self.forward_attnpool(aggregated)


def forward_clip_snn_diff_leaky_rate_s(self, x):
    backbone_outputs = []
    backbone_outputs.append(self.init_forward_backbone(torch.zeros_like(x)))
    backbone_outputs.append(self.init_forward_backbone(x))
    mul = 1
    for _ in range(1, self.T):
        mul /= self.tau
        backbone_outputs.append(self.init_forward_backbone(torch.zeros_like(x)))
    aggregated = decodeoutput(torch.stack(backbone_outputs, dim=0)).mean(dim=0)
    return self.forward_attnpool(aggregated)


def forward_replace_clip(args, model):
    model.coding_type = args.coding_type
    model.step_mode = args.step_mode
    model.convert_attn = getattr(args, 'convert_attn', False)
    model.init_forward_backbone = model.forward_backbone

    convert_attn = model.convert_attn
    if convert_attn:
        model.init_forward = model.forward

    if args.coding_type == 'rate':
        if args.step_mode == 's':
            model.T = args.time
            if convert_attn:
                model.forward = types.MethodType(forward_clip_snn_rate_s_full, model)
            else:
                model.forward = types.MethodType(forward_clip_snn_rate_s, model)
        elif args.step_mode == 'm':
            model.T = args.time
            model.merge = MergeTemporalDim()
            model.expand = ExpandTemporalDim(model.T)
            if convert_attn:
                model.forward = types.MethodType(forward_clip_snn_rate_m_full, model)
            else:
                model.forward = types.MethodType(forward_clip_snn_rate_m, model)
    elif args.coding_type == 'leaky_rate':
        model.tau = args.tau
        if args.step_mode == 's':
            model.T = args.time
            if convert_attn:
                model.forward = types.MethodType(forward_clip_snn_leaky_rate_s_full, model)
            else:
                model.forward = types.MethodType(forward_clip_snn_leaky_rate_s, model)
        elif args.step_mode == 'm':
            model.T = args.time
            model.merge = MergeTemporalDim()
            model.expand = ExpandTemporalDim(model.T)
            if convert_attn:
                model.forward = types.MethodType(forward_clip_snn_leaky_rate_m_full, model)
            else:
                model.forward = types.MethodType(forward_clip_snn_leaky_rate_m, model)
    elif args.coding_type == 'diff_rate':
        if args.step_mode == 's':
            model.T = args.time
            if convert_attn:
                model.forward = types.MethodType(forward_clip_snn_diff_rate_s_full, model)
            else:
                model.forward = types.MethodType(forward_clip_snn_diff_rate_s, model)
        elif args.step_mode == 'm':
            model.T = args.time
            model.merge = MergeTemporalDim()
            model.expand = ExpandTemporalDim(model.T + 1)
            if convert_attn:
                model.forward = types.MethodType(forward_clip_snn_diff_rate_m_full, model)
            else:
                model.forward = types.MethodType(forward_clip_snn_diff_rate_m, model)
    elif args.coding_type == 'diff_leaky_rate':
        model.tau = args.tau
        if args.step_mode == 's':
            model.T = args.time
            if convert_attn:
                model.forward = types.MethodType(forward_clip_snn_diff_leaky_rate_s_full, model)
            else:
                model.forward = types.MethodType(forward_clip_snn_diff_leaky_rate_s, model)
        elif args.step_mode == 'm':
            model.T = args.time
            model.merge = MergeTemporalDim()
            model.expand = ExpandTemporalDim(model.T)
            if convert_attn:
                model.forward = types.MethodType(forward_clip_snn_diff_leaky_rate_m_full, model)
            else:
                model.forward = types.MethodType(forward_clip_snn_diff_leaky_rate_m, model)
    else:
        raise ValueError(f"Unexpected coding_type: {args.coding_type}")

    return model
