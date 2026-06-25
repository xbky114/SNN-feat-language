import argparse
import os
import sys

import torch
import yaml

DCGS_ROOT = os.path.dirname(os.path.abspath(__file__))
if DCGS_ROOT not in sys.path:
    sys.path.insert(0, DCGS_ROOT)

CLIP_ROOT = os.path.abspath(os.path.join(DCGS_ROOT, '..', 'CLIP'))
if CLIP_ROOT not in sys.path:
    sys.path.insert(0, CLIP_ROOT)

from converter import Converter, Threshold_Getter
from forwards import forward_replace
from utils.clip_weights import extract_visual_state_dict
from main import load_model_from_dict
from models import modelpool
from models.clip_wrapper import CLIPWithSNNVisual
from train_val_functions.val_functions import val_snn_clip
from utils import reset, seed_all


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/clip/rn50_config_test_channel_mth.yaml', type=str)
    parser.add_argument('--device', default='0', type=str)
    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as file:
        config = yaml.safe_load(file)

    class Args:
        pass

    args = Args()
    for key, value in config.items():
        setattr(args, key, value)
    args.device = cli_args.device
    return args


def build_snn_visual(args, device):
    model = modelpool(args)
    model.convert_attn = getattr(args, 'convert_attn', False)
    model = Converter.change_maxpool_before_relu(model)
    model = Converter.replace_by_maxpool_neuron(
        model, T=args.time, step_mode=args.step_mode, coding_type=args.coding_type
    )
    model = Threshold_Getter.replace_nonlinear_by_hook(
        model=model,
        momentum=0.1,
        mode=args.threshold_mode,
        level=args.threshold_level,
        convert_attn=getattr(args, 'convert_attn', False),
    )
    model = load_model_from_dict(model, args.load_name, device, model_name=args.model_name)
    if args.model_name == 'clip_rn50' and not getattr(args, 'convert_attn', False):
        clip_path = getattr(args, 'clip_checkpoint', None) or os.path.expanduser('~/.cache/clip/RN50.pt')
        if os.path.exists(os.path.expanduser(clip_path)):
            attnpool_state = {
                key: value.float()
                for key, value in extract_visual_state_dict(clip_path).items()
                if key.startswith('attnpool.')
            }
            model.load_state_dict(attnpool_state, strict=False)
    if args.threshold_mode == "var":
        if args.neuron_name.startswith('MTH'):
            model = Threshold_Getter.get_scale_from_var(model, T=args.time * (2 ** args.num_thresholds))
        else:
            model = Threshold_Getter.get_scale_from_var(model, T=args.time)
    model_converter = Converter(
        neuron=args.neuron_name,
        args=args,
        T=args.time,
        step_mode=args.step_mode,
        fuse_flag=getattr(args, 'fuse', False),
    )
    model = model_converter(model)
    model = forward_replace(args, model)
    model.to(device)
    model.eval()
    return model


def main():
    args = load_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(getattr(args, 'seed', 2024))

    import clip
    from PIL import Image

    clip_checkpoint = getattr(args, 'clip_checkpoint', None) or os.path.expanduser('~/.cache/clip/RN50.pt')
    ann_model, preprocess = clip.load("RN50", device=device, jit=False, download_root=os.path.expanduser('~/.cache/clip'))
    if clip_checkpoint and os.path.exists(clip_checkpoint) and not str(clip_checkpoint).endswith('.pth'):
        ann_model, preprocess = clip.load(clip_checkpoint, device=device, jit=False)
    ann_model = ann_model.float()

    snn_visual = build_snn_visual(args, device)
    wrapper = CLIPWithSNNVisual(snn_visual, ann_model).to(device).eval()

    clip_image_path = getattr(args, 'clip_image_path', os.path.join(CLIP_ROOT, 'CLIP.png'))
    labels = getattr(args, 'clip_labels', ["a diagram", "a dog", "a cat"])
    image = preprocess(Image.open(clip_image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        probs_ann = ann_model(image, text)[0].softmax(dim=-1).cpu().numpy()
        reset(snn_visual)
        probs_snn = wrapper(image, text)[0].softmax(dim=-1).cpu().numpy()

    print("Labels:", labels)
    print("ANN probs:", probs_ann)
    print("SNN probs:", probs_snn)
    print("Abs diff:", abs(probs_ann - probs_snn))

    val_snn_clip(snn_visual, None, device, args)


if __name__ == "__main__":
    main()
