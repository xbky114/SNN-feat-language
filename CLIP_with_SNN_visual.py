import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DCGS_ROOT = os.path.join(_REPO_ROOT, "DCGS")
if _DCGS_ROOT not in sys.path:
    sys.path.insert(0, _DCGS_ROOT)


def _build_args(
    convert_attn: bool,
    thre_path: str,
    T: int,
    clip_checkpoint: str = None,
    neuron_name: str = "IF",
    num_thresholds: int = 1,
):
    if neuron_name not in ("IF", "MTH"):
        raise ValueError(f"neuron_name must be 'IF' or 'MTH', got {neuron_name!r}")
    if num_thresholds < 1:
        raise ValueError(f"num_thresholds must be >= 1, got {num_thresholds}")

    if convert_attn:
        threshold_mode = "99.9%"
        c = 3.0
    else:
        threshold_mode = "var"
        c = 1.0

    return SimpleNamespace(
        model_name="clip_rn50",
        load_name=os.path.expanduser(thre_path),
        mode="test_snn",
        task="clip",
        threshold_mode=threshold_mode,
        threshold_level="channel",
        neuron_name=neuron_name,
        num_thresholds=num_thresholds,
        step_mode="m",
        coding_type="rate",
        fuse=False,
        convert_attn=convert_attn,
        c=c,
        time=T,
        clip_checkpoint=clip_checkpoint or os.path.expanduser("~/.cache/clip/RN50.pt"),
    )


def load_converted_snn(
    convert_attn: bool,
    thre_path: str,
    T: int,
    device,
    clip_checkpoint: str = None,
    neuron_name: str = "IF",
    num_thresholds: int = 1,
) -> nn.Module:
    """Load CLIP RN50 visual SNN. T is simulation steps; forward returns [T, B, 1024]."""
    from converter import Converter, Threshold_Getter
    from forwards import forward_replace
    from main import load_model_from_dict
    from models import modelpool
    from utils.clip_weights import extract_visual_state_dict

    if isinstance(device, str):
        device = torch.device(device)

    args = _build_args(
        convert_attn, thre_path, T, clip_checkpoint, neuron_name, num_thresholds
    )

    model = modelpool(args)
    model.convert_attn = convert_attn
    model = Converter.change_maxpool_before_relu(model)
    model = Converter.replace_by_maxpool_neuron(
        model, T=T, step_mode=args.step_mode, coding_type=args.coding_type
    )
    model = Threshold_Getter.replace_nonlinear_by_hook(
        model=model,
        momentum=0.1,
        mode=args.threshold_mode,
        level=args.threshold_level,
        convert_attn=convert_attn,
    )
    model = load_model_from_dict(model, args.load_name, device, model_name=args.model_name)

    if args.model_name == "clip_rn50" and not convert_attn:
        clip_path = args.clip_checkpoint
        if os.path.exists(os.path.expanduser(clip_path)):
            attnpool_state = {
                key: value.float()
                for key, value in extract_visual_state_dict(clip_path).items()
                if key.startswith("attnpool.")
            }
            model.load_state_dict(attnpool_state, strict=False)

    if args.threshold_mode == "var":
        if args.neuron_name.startswith("MTH"):
            model = Threshold_Getter.get_scale_from_var(
                model, T=T * (2 ** args.num_thresholds)
            )
        else:
            model = Threshold_Getter.get_scale_from_var(model, T=T)

    model_converter = Converter(
        neuron=args.neuron_name,
        args=args,
        T=T,
        step_mode=args.step_mode,
        fuse_flag=args.fuse,
    )
    model = model_converter(model)
    model = forward_replace(args, model)
    model.to(device)
    model.eval()
    return model


def wrap_snn_as_clip(snn_visual, clip_checkpoint=None, device=None):
    """Wrap SNN visual encoder with CLIP text tower. Exposes encode_image/encode_text/forward/test_probs_at_all_t."""
    import clip

    clip_root = os.path.join(_REPO_ROOT, "CLIP")
    if clip_root not in sys.path:
        sys.path.insert(0, clip_root)

    from models.clip_wrapper import CLIPWithSNNVisual

    if device is None:
        device = next(snn_visual.parameters()).device
    clip_checkpoint = clip_checkpoint or os.path.expanduser("~/.cache/clip/RN50.pt")
    clip_model, _ = clip.load(
        "RN50", device=device, jit=False, download_root=os.path.expanduser("~/.cache/clip")
    )
    if clip_checkpoint and os.path.exists(os.path.expanduser(clip_checkpoint)):
        if not str(clip_checkpoint).endswith(".pth"):
            clip_model, _ = clip.load(
                os.path.expanduser(clip_checkpoint), device=device, jit=False
            )
    clip_model = clip_model.float()
    return CLIPWithSNNVisual(snn_visual, clip_model).to(device).eval()


__all__ = ["load_converted_snn", "wrap_snn_as_clip"]
