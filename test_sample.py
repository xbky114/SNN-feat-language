#!/usr/bin/env python3
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_CLIP_ROOT = os.path.join(_REPO_ROOT, "CLIP")
if _CLIP_ROOT not in sys.path:
    sys.path.insert(0, _CLIP_ROOT)

from CLIP_with_SNN_visual import load_converted_snn, wrap_snn_as_clip

_DCGS_ROOT = os.path.join(_REPO_ROOT, "DCGS")
if _DCGS_ROOT not in sys.path:
    sys.path.insert(0, _DCGS_ROOT)
from utils import reset


def parse_args():
    parser = argparse.ArgumentParser(description="Single-sample SNN CLIP evaluation")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "descriptions",
        nargs="+",
        type=str,
        help="Text labels for zero-shot matching",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/SNN_CLIP/",
        help="Directory to save json and plot",
    )
    parser.add_argument(
        "--thre-path",
        type=str,
        default="DCGS/output/threshold_clip_rn50_channel_ann_attn.pth",
        help="Threshold checkpoint from get_threshold",
    )
    parser.add_argument("--T", type=int, default=256, help="SNN simulation steps")
    parser.add_argument(
        "--convert-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convert attention pool to SNN",
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=str,
        default=os.path.expanduser("~/.cache/clip/RN50.pt"),
        help="CLIP RN50 checkpoint for ANN baseline and text tower",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    return parser.parse_args()


@torch.no_grad()
def run_eval(args):
    import clip

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    snn_visual = load_converted_snn(
        args.convert_attn,
        args.thre_path,
        args.T,
        device,
        clip_checkpoint=args.clip_checkpoint,
    )
    wrapper = wrap_snn_as_clip(
        snn_visual,
        clip_checkpoint=args.clip_checkpoint,
        device=device,
    )

    ann_model, preprocess = clip.load(
        "RN50",
        device=device,
        jit=False,
        download_root=os.path.expanduser("~/.cache/clip"),
    )
    ckpt = args.clip_checkpoint
    if ckpt and os.path.exists(os.path.expanduser(ckpt)) and not ckpt.endswith(".pth"):
        ann_model, _ = clip.load(os.path.expanduser(ckpt), device=device, jit=False)
    ann_model = ann_model.float().eval()

    image = preprocess(Image.open(args.image)).unsqueeze(0).to(device)
    text = clip.tokenize(args.descriptions).to(device)
    labels = list(args.descriptions)

    probs_ann = ann_model(image, text)[0].softmax(dim=-1).cpu().numpy()

    reset(snn_visual)
    img_features = wrapper.encode_image(image)
    text_features = wrapper.encode_text(text)
    probs_snn = (
        wrapper.forward(img_features[-1], text_features)[0]
        .softmax(dim=-1)
        .cpu()
        .numpy()
    )

    reset(snn_visual)
    probs_all_t = wrapper.test_probs_at_all_t(image, text)

    result = {
        "image": os.path.abspath(args.image),
        "labels": labels,
        "ann_probs": probs_ann[0].tolist(),
        "snn_probs": probs_snn[0].tolist(),
        "ann_top1": labels[int(probs_ann[0].argmax())],
        "snn_top1": labels[int(probs_snn[0].argmax())],
    }
    json_path = os.path.join(args.output_dir, "probs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    probs_t = probs_all_t[0]
    T = probs_t.shape[0]
    steps = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, label in enumerate(labels):
        ax.plot(steps, probs_t[:, i], label=label, linewidth=1.5)
    ax.set_xlabel("time step")
    ax.set_ylabel("prob")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("SNN probs over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "probs_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved:", json_path)
    print("Saved:", plot_path)


if __name__ == "__main__":
    run_eval(parse_args())
