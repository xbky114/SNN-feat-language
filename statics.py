#!/usr/bin/env python3
"""Aggregate dataset-wide SNN CLIP metrics from per-image metrics.json files."""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute dataset-wide metrics from per-image metrics.json files."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="results/CLIP/auxilary_2000/CLIP",
        help="Root directory with two-level subfolders (each leaf folder is one image).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save JSON and plots (default: <data-root>/statics).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of relative-depth bins M for Temporal COM metrics.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Small constant for numerical stability.",
    )
    return parser.parse_args()


def iter_sample_dirs(data_root):
    """Yield leaf sample directories under two nested folder levels."""
    for level1 in sorted(os.listdir(data_root)):
        path1 = os.path.join(data_root, level1)
        if not os.path.isdir(path1):
            continue
        for level2 in sorted(os.listdir(path1)):
            path2 = os.path.join(path1, level2)
            if os.path.isdir(path2):
                yield path2


def load_sample_metrics(sample_dir):
    json_path = os.path.join(sample_dir, "metrics.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    probs = np.asarray(data["snn_all_t"]["probs"], dtype=np.float64)
    logits = np.asarray(data["snn_all_t"]["logits"], dtype=np.float64)
    labels = data["labels"]
    if probs.ndim != 2 or logits.ndim != 2:
        raise ValueError(f"Expected 2D time series in {json_path}")
    if probs.shape != logits.shape:
        raise ValueError(f"probs/logits shape mismatch in {json_path}")
    if probs.shape[1] != len(labels):
        raise ValueError(f"Label count mismatch in {json_path}")
    return {
        "sample_dir": sample_dir,
        "labels": labels,
        "probs": probs,
        "logits": logits,
        "T": probs.shape[0],
        "n": probs.shape[1],
    }


def relative_depths(n):
    """Return D_k for k=1..n mapped to 0-indexed array entries."""
    if n == 1:
        return np.array([1.0], dtype=np.float64)
    return np.arange(n, dtype=np.float64) / (n - 1)


def expected_depth_sequence(probs_tn):
    """Compute E[D_t] for each time step from softmax probabilities."""
    depths = relative_depths(probs_tn.shape[1])
    return probs_tn @ depths


def time_centroid(values_t, normalize, eps):
    """Compute temporal center of mass for one label node."""
    values_t = np.asarray(values_t, dtype=np.float64)
    if normalize:
        vmin = values_t.min()
        vmax = values_t.max()
        normed = (values_t - vmin) / (vmax - vmin + eps)
    else:
        normed = values_t
    total = normed.sum()
    weights = normed / (total + eps)
    times = np.arange(1, values_t.shape[0] + 1, dtype=np.float64)
    return float(times @ weights)


def depth_bin_index(k_index, n, num_bins):
    """Map 0-based label index to discrete depth bin m in [1, M]."""
    if n == 1:
        d_rel = 1.0
    else:
        d_rel = k_index / (n - 1)
    return int(round(d_rel * (num_bins - 1)) + 1)


def spearman_result(x, y):
    rho, p_value = spearmanr(x, y)
    return {"spearman_r": float(rho), "spearman_p": float(p_value)}


def plot_scatter(x, y, xlabel, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=28, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(data_root, "statics")
    )
    os.makedirs(output_dir, exist_ok=True)

    expected_depths = []
    logit_bins = [[] for _ in range(args.num_bins + 1)]
    prob_bins = [[] for _ in range(args.num_bins + 1)]
    num_samples = 0
    skipped = 0

    for sample_dir in iter_sample_dirs(data_root):
        sample = load_sample_metrics(sample_dir)
        if sample is None:
            skipped += 1
            continue

        num_samples += 1
        probs = sample["probs"]
        logits = sample["logits"]
        n = sample["n"]
        expected_depths.append(expected_depth_sequence(probs))

        for k in range(n):
            bin_idx = depth_bin_index(k, n, args.num_bins)
            logit_bins[bin_idx].append(
                time_centroid(logits[:, k], normalize=True, eps=args.epsilon)
            )
            prob_bins[bin_idx].append(
                time_centroid(probs[:, k], normalize=False, eps=args.epsilon)
            )

        if num_samples % 200 == 0:
            print(f"Processed {num_samples} samples...", flush=True)

    if num_samples == 0:
        print("No metrics.json files found.", file=sys.stderr)
        sys.exit(1)

    expected_depths = np.stack(expected_depths, axis=0)
    mean_expected_depth = expected_depths.mean(axis=0)
    time_steps = np.arange(1, expected_depths.shape[1] + 1, dtype=np.float64)

    mean_level_corr = spearman_result(time_steps, mean_expected_depth)

    def aggregate_com_bins(bins):
        valid_x = []
        valid_y = []
        bin_counts = {}
        for m in range(1, args.num_bins + 1):
            values = bins[m]
            bin_counts[str(m)] = len(values)
            if values:
                valid_x.append(m)
                valid_y.append(float(np.mean(values)))
        corr = spearman_result(valid_x, valid_y) if len(valid_x) >= 2 else {
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }
        return {
            **corr,
            "depth_bins": valid_x,
            "mean_time_centroid": valid_y,
            "bin_counts": bin_counts,
        }

    logit_com = aggregate_com_bins(logit_bins)
    prob_com = aggregate_com_bins(prob_bins)

    result = {
        "data_root": data_root,
        "num_samples": num_samples,
        "num_skipped": skipped,
        "num_bins": args.num_bins,
        "epsilon": args.epsilon,
        "T": int(expected_depths.shape[1]),
        "metrics": {
            "mean_level_correlation": {
                **mean_level_corr,
                "time_steps": time_steps.tolist(),
                "mean_expected_depth": mean_expected_depth.tolist(),
            },
            "temporal_logits_com": logit_com,
            "temporal_prob_com": prob_com,
        },
    }

    json_path = os.path.join(output_dir, "dataset_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    plot_scatter(
        time_steps,
        mean_expected_depth,
        xlabel="time step t",
        ylabel=r"$\bar{E}[D_t]$",
        title=f"Mean Level Correlation (r={mean_level_corr['spearman_r']:.4f})",
        save_path=os.path.join(output_dir, "mean_level_correlation_scatter.png"),
    )
    plot_scatter(
        logit_com["depth_bins"],
        logit_com["mean_time_centroid"],
        xlabel="depth bin m",
        ylabel=r"$\bar{T}_m$",
        title=f"Temporal Logits COM (r={logit_com['spearman_r']:.4f})",
        save_path=os.path.join(output_dir, "temporal_logits_com_scatter.png"),
    )
    plot_scatter(
        prob_com["depth_bins"],
        prob_com["mean_time_centroid"],
        xlabel="depth bin m",
        ylabel=r"$\bar{T}_m$",
        title=f"Temporal Prob COM (r={prob_com['spearman_r']:.4f})",
        save_path=os.path.join(output_dir, "temporal_prob_com_scatter.png"),
    )

    print(f"Samples processed: {num_samples} (skipped: {skipped})")
    print(f"mean_level_correlation spearman_r = {mean_level_corr['spearman_r']:.6f}")
    print(f"temporal_logits_com      spearman_r = {logit_com['spearman_r']:.6f}")
    print(f"temporal_prob_com        spearman_r = {prob_com['spearman_r']:.6f}")
    print(f"Saved: {json_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
