"""Reject unsupported registered modules before calibration or conversion."""

from torch import nn


def reject_unsupported_modules(model, method):
    forbidden = (
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        nn.FractionalMaxPool2d, nn.FractionalMaxPool3d,
        nn.LayerNorm, nn.GELU, nn.Softmax, nn.Softmax2d, nn.LogSoftmax,
        nn.MultiheadAttention,
    )
    for path, module in model.named_modules():
        # CLIP uses custom QuickGELU/LayerNorm/AttentionPool classes. Check
        # their MRO too so subclasses cannot hide these known operations.
        names = [cls.__name__.lower().replace("_", "") for cls in type(module).__mro__]
        custom = any(any(token in name for token in
                         ("maxpool", "layernorm", "gelu", "softmax", "attention", "attnpool"))
                     or name.endswith("attn") or name == "myat" for name in names)
        if isinstance(module, forbidden) or custom:
            raise ValueError(f"{method} does not support {type(module).__name__} at {path or '<root>'!r}")
