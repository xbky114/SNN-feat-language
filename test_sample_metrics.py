#!/usr/bin/env python3
import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Single-sample SNN CLIP metrics over time")
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
        default="./results/SNN_CLIP_metrics/",
        help="Directory to save json, csv, and plots",
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
    parser.add_argument(
        "--neuron-name",
        type=str,
        choices=["IF", "MTH"],
        default="IF",
        help="SNN neuron type",
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=1,
        help="Number of thresholds for MTH neuron",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    return parser.parse_args()


def l2_normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def plot_metric(steps, values_tn, labels, ylabel, title, save_path, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, label in enumerate(labels):
        ax.plot(steps, values_tn[:, i], label=label, linewidth=1.5)
    ax.set_xlabel("time step")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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
        neuron_name=args.neuron_name,
        num_thresholds=args.num_thresholds,
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

    image = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    text = clip.tokenize(args.descriptions).to(device)
    labels = list(args.descriptions)

    ann_logits = ann_model(image, text)[0].float()
    ann_probs = ann_logits.softmax(dim=-1)

    ann_img_feat = ann_model.encode_image(image).float()
    ann_text_feat = ann_model.encode_text(text).float()
    ann_cosine = l2_normalize(ann_img_feat) @ l2_normalize(ann_text_feat).t()

    reset(snn_visual)
    img_features_all_t = wrapper.encode_image(image).float()
    text_features = wrapper.encode_text(text).float()

    img_features_norm = l2_normalize(img_features_all_t)
    text_features_norm = l2_normalize(text_features)

    cosine_all_t = torch.einsum("tbd,nd->btn", img_features_norm, text_features_norm)

    logit_scale = wrapper.logit_scale.exp().float()
    logits_all_t = cosine_all_t * logit_scale
    probs_all_t = logits_all_t.softmax(dim=-1)

    probs_t = probs_all_t[0].cpu().numpy()
    logits_t = logits_all_t[0].cpu().numpy()
    cosine_t = cosine_all_t[0].cpu().numpy()

    final_probs = probs_t[-1]
    final_logits = logits_t[-1]
    final_cosine = cosine_t[-1]

    result = {
        "image": os.path.abspath(args.image),
        "labels": labels,
        "neuron_name": args.neuron_name,
        "num_thresholds": args.num_thresholds,
        "T": args.T,
        "convert_attn": args.convert_attn,
        "logit_scale": float(logit_scale.cpu()),
        "ann": {
            "probs": ann_probs[0].cpu().tolist(),
            "logits": ann_logits[0].cpu().tolist(),
            "cosine": ann_cosine[0].cpu().tolist(),
            "top1_prob": labels[int(ann_probs[0].argmax().cpu())],
            "top1_logit": labels[int(ann_logits[0].argmax().cpu())],
            "top1_cosine": labels[int(ann_cosine[0].argmax().cpu())],
        },
        "snn_final": {
            "probs": final_probs.tolist(),
            "logits": final_logits.tolist(),
            "cosine": final_cosine.tolist(),
            "top1_prob": labels[int(final_probs.argmax())],
            "top1_logit": labels[int(final_logits.argmax())],
            "top1_cosine": labels[int(final_cosine.argmax())],
        },
        "snn_all_t": {
            "probs": probs_t.tolist(),
            "logits": logits_t.tolist(),
            "cosine": cosine_t.tolist(),
        },
    }

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output_dir, "metrics_over_time.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_step", "label", "softmax_prob", "logit", "cosine"])
        for t in range(probs_t.shape[0]):
            for i, label in enumerate(labels):
                writer.writerow([
                    t + 1,
                    label,
                    float(probs_t[t, i]),
                    float(logits_t[t, i]),
                    float(cosine_t[t, i]),
                ])

    steps = np.arange(1, probs_t.shape[0] + 1)

    probs_plot_path = os.path.join(args.output_dir, "probs_curve.png")
    logits_plot_path = os.path.join(args.output_dir, "logits_curve.png")
    cosine_plot_path = os.path.join(args.output_dir, "cosine_curve.png")

    plot_metric(
        steps,
        probs_t,
        labels,
        ylabel="softmax probability",
        title="SNN softmax probabilities over time",
        save_path=probs_plot_path,
        ylim=(0.0, 1.0),
    )
    plot_metric(
        steps,
        logits_t,
        labels,
        ylabel="CLIP logit",
        title="SNN raw logits over time",
        save_path=logits_plot_path,
    )
    plot_metric(
        steps,
        cosine_t,
        labels,
        ylabel="cosine similarity",
        title="SNN cosine similarities over time",
        save_path=cosine_plot_path,
    )

    summary = {
        "image": result["image"],
        "labels": labels,
        "ann": result["ann"],
        "snn_final": result["snn_final"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved:", json_path)
    print("Saved:", csv_path)
    print("Saved:", probs_plot_path)
    print("Saved:", logits_plot_path)
    print("Saved:", cosine_plot_path)


if __name__ == "__main__":
    run_eval(parse_args())