#!/usr/bin/env python3
"""Measure normalized-feature value masking's effect on coarse-vs-fine CLIP logit margins.

The validation directory is expected to have ImageNet's usual layout:
``val/n########/image.JPEG``.  A directory name is interpreted as the WordNet
noun synset offset.  For each image, only the terms on that synset's selected
hypernym path (excluding WordNet's shared top root) are candidate texts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from wordnet_geometry import (
    Synset,
    leaf_paths,
    parse_data_noun,
    parse_index_noun,
    parse_noun_senses,
    validate_indexes,
)


@dataclass(frozen=True)
class Sample:
    image: Path
    leaf: str
    path: tuple[str, ...]


def load_wordnet(wordnet_dir: Path) -> tuple[dict[str, Synset], str]:
    """Read and cross-check exactly the three requested WordNet noun files."""
    required = {name: wordnet_dir / name for name in ("data.noun", "index.noun", "index.sense")}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing WordNet files: " + ", ".join(missing))
    synsets = parse_data_noun(required["data.noun"])
    noun_index = parse_index_noun(required["index.noun"])
    noun_senses = parse_noun_senses(required["index.sense"])
    validate_indexes(synsets, noun_index, noun_senses)
    root, _ = leaf_paths(synsets)  # also validates the single-root hierarchy
    return synsets, root


def selected_path(leaf: str, synsets: dict[str, Synset], root: str) -> tuple[str, ...]:
    """Follow the first @/@i parent, deliberately ignoring multiple inheritance."""
    reverse: list[str] = []
    seen: set[str] = set()
    current: str | None = leaf
    while current is not None:
        if current in seen:
            raise ValueError(f"Cycle while following ImageNet synset {leaf}")
        if current not in synsets:
            raise KeyError(f"ImageNet synset {leaf} is not in data.noun")
        seen.add(current)
        reverse.append(current)
        current = synsets[current].parent
    path = tuple(reversed(reverse))
    if path[0] != root or len(path) < 2:
        raise ValueError(f"ImageNet synset {leaf} does not have a usable path below {root}")
    return path[1:]


def discover_samples(val_dir: Path, synsets: dict[str, Synset], root: str, max_images: int | None) -> list[Sample]:
    extensions = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
    samples: list[Sample] = []
    class_dirs = sorted(path for path in val_dir.iterdir() if path.is_dir())
    for class_dir in tqdm(class_dirs, desc="Scanning validation set", unit="class"):
        # ImageNet folders spell a WordNet noun offset as n########; data.noun
        # itself stores just the eight digits.
        folder_label = class_dir.name
        leaf = folder_label[1:] if len(folder_label) == 9 and folder_label.startswith("n") and folder_label[1:].isdigit() else folder_label
        path = selected_path(leaf, synsets, root)
        for image in sorted(class_dir.iterdir()):
            if image.is_file() and image.suffix.lower() in extensions:
                samples.append(Sample(image, leaf, path))
                if max_images is not None and len(samples) >= max_images:
                    return samples
    if not samples:
        raise ValueError(f"No images found under {val_dir}")
    return samples


def load_clip(clip_dir: Path, device: str):
    import torch

    if device != "cuda:0":
        raise ValueError("This experiment is fixed to --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cuda:0 is required")
    sys.path.insert(0, str(clip_dir))
    import clip  # type: ignore

    model, preprocess = clip.load("RN50", device=device, jit=False)
    model.eval()
    return model, preprocess, clip


def batches(items: list[Sample], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def image_features(model, preprocess, items: list[Sample], device: str, batch_size: int, description: str):
    """Yield raw, *not normalized*, image-tower outputs and their samples."""
    import torch
    from PIL import Image

    with torch.inference_mode():
        with tqdm(total=len(items), desc=description, unit="image") as progress:
            for _, batch in batches(items, batch_size):
                images = []
                for sample in batch:
                    with Image.open(sample.image) as image:
                        images.append(preprocess(image.convert("RGB")))
                tensor = torch.stack(images).to(device, non_blocking=True)
                features = model.encode_image(tensor).float()
                progress.update(len(batch))
                yield features, batch


def calculate_threshold(model, preprocess, text_by_word, synsets, samples: list[Sample], args: argparse.Namespace) -> tuple[float, dict[str, float | int]]:
    """Calculate the requested percentile from normalized features of correct images only."""
    import torch
    from numpy.lib.format import open_memmap

    print("Calculating normalized-feature threshold (collecting absolute values)...", flush=True)
    normalized_values = None
    position = 0
    correct_images = 0
    for raw, batch in image_features(model, preprocess, samples, args.device, args.batch_size, "Threshold pass"):
        unit = raw / raw.norm(dim=-1, keepdim=True)
        correct_indices = []
        for index, sample in enumerate(batch):
            candidate_features = coarse_and_fine_features(sample, synsets, text_by_word, args.coarse)
            candidate_logits = scores(unit[index], candidate_features, model)
            if candidate_logits[1] > candidate_logits[0]:
                correct_indices.append(index)
        if not correct_indices:
            continue
        normalized_cpu = unit[correct_indices].cpu().numpy()
        if normalized_values is None:
            shape = (len(samples) * normalized_cpu.shape[1],)
            normalized_values = open_memmap(args.output_dir / ".absolute_normalized_image_features.tmp.npy", mode="w+", dtype=np.float32, shape=shape)
        end = position + normalized_cpu.size
        normalized_values[position:end] = np.abs(normalized_cpu).reshape(-1)
        position = end
        correct_images += len(correct_indices)
    if normalized_values is None:
        raise ValueError("No images had l_fine > l_coarse in the two-text candidate set")
    normalized_values.flush()
    print(f"Computing exact normalized-feature {args.percentile:g}th percentile...", flush=True)
    threshold = float(np.percentile(normalized_values[:position], args.percentile))
    # Persist exactly the retained values, without unused zero-filled capacity.
    np.save(args.output_dir / "absolute_normalized_image_features.npy", normalized_values[:position])
    del normalized_values
    (args.output_dir / ".absolute_normalized_image_features.tmp.npy").unlink()
    print(f"Threshold ready: normalized={threshold:.8g}", flush=True)
    accuracy = correct_images / len(samples)
    print(f"Original two-text accuracy (l_fine > l_coarse): {correct_images}/{len(samples)} ({accuracy:.6%})", flush=True)
    return threshold, {"total_images": len(samples), "correct_images": correct_images, "accuracy": accuracy}


def encode_texts(model, clip_module, synsets: dict[str, Synset], samples: list[Sample], args: argparse.Namespace):
    import torch

    # First data.noun lemma verbatim: no article, template, or other prompt wrapper.
    offsets = {
        offset
        for sample in samples
        for offset in (sample.path[0] if args.coarse == "root" else sample.path[len(sample.path) // 2], sample.path[-1])
    }
    texts = sorted(synsets[offset].words[0] for offset in offsets)
    print(f"Encoding {len(texts)} unique WordNet texts with RN50...", flush=True)
    encoded: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.text_batch_size), desc="Text encoding", unit="batch"):
            tokens = clip_module.tokenize(texts[start : start + args.text_batch_size]).to(args.device)
            feature = model.encode_text(tokens).float()
            encoded.append(feature / feature.norm(dim=-1, keepdim=True))
    return {text: value for text, value in zip(texts, torch.cat(encoded, dim=0))}


def scores(image_feature, text_features, model):
    import torch

    matrix = torch.stack(text_features)
    return model.logit_scale.exp().float() * image_feature @ matrix.T


def coarse_and_fine_features(sample: Sample, synsets: dict[str, Synset], text_by_word, coarse: str):
    """Return the two and only two candidate text features: y_coarse and y_fine."""
    coarse_offset = sample.path[0] if coarse == "root" else sample.path[len(sample.path) // 2]
    fine_offset = sample.path[-1]
    coarse_text = synsets[coarse_offset].words[0]
    fine_text = synsets[fine_offset].words[0]
    return [text_by_word[coarse_text], text_by_word[fine_text]]


def render_margin_histogram(output_dir: Path, margin_changes: list[float], title: str) -> dict[str, float | int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    margin_array = np.asarray(margin_changes)
    mean = float(margin_array.mean()) if len(margin_array) else float("nan")
    probability_negative = float((margin_array < 0).mean()) if len(margin_array) else float("nan")
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.hist(margin_array, bins="auto", color="#4477aa", edgecolor="white")
    axis.axvline(0, color="black", linestyle="--", label="zero")
    axis.axvline(mean, color="#cc3311", label=f"mean = {mean:.6g}")
    axis.text(0.98, 0.96, f"P(ΔM < 0) = {probability_negative:.2%}", transform=axis.transAxes, ha="right", va="top")
    axis.set(xlabel=r"$\Delta M=M_{masked}-M_{original}$", ylabel="images", title=title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "logit_margin_change_histogram.png", dpi=180)
    plt.close(fig)
    return {"correct_images": len(margin_changes), "mean_delta_margin": mean, "P_delta_margin_lt_0": probability_negative}


def render_logit_scatter(output_dir: Path, original_coarse: list[float], original_fine: list[float], clipped_coarse: list[float], clipped_fine: list[float], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_coarse_array = np.asarray(original_coarse)
    original_fine_array = np.asarray(original_fine)
    clipped_coarse_array = np.asarray(clipped_coarse)
    clipped_fine_array = np.asarray(clipped_fine)
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(original_coarse_array, original_fine_array, s=5, alpha=0.35, color="#4477aa", label="original", rasterized=True)
    axis.scatter(clipped_coarse_array, clipped_fine_array, s=5, alpha=0.35, color="#cc3311", label="masked", rasterized=True)
    all_values = np.concatenate((original_coarse_array, original_fine_array, clipped_coarse_array, clipped_fine_array))
    low, high = (float(all_values.min()), float(all_values.max())) if len(all_values) else (-1.0, 1.0)
    axis.plot([low, high], [low, high], "k--", linewidth=1, label="y=x")
    axis.set(xlabel=r"$l_{coarse}$", ylabel=r"$l_{fine}$", title=title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "coarse_vs_fine_logits_scatter.png", dpi=180)
    plt.close(fig)


def evaluate(threshold, model, preprocess, text_by_word, synsets, samples, args):
    import torch

    original_margins: list[float] = []
    clipped_margins: list[float] = []
    margin_changes: list[float] = []
    original_coarse_logits: list[float] = []
    original_fine_logits: list[float] = []
    clipped_coarse_logits: list[float] = []
    clipped_fine_logits: list[float] = []
    newly_correct_images = 0
    print(f"Evaluating normalized-feature value masking (threshold={threshold:.8g}, coarse={args.coarse})...", flush=True)
    with torch.inference_mode():
        for raw, batch in image_features(model, preprocess, samples, args.device, args.batch_size, "Clipping evaluation"):
            unit = raw / raw.norm(dim=-1, keepdim=True)
            # Preserve values at or above the threshold and zero smaller magnitudes.
            # The normalized feature is not re-normalized after this operation.
            altered = torch.where(unit.abs() < threshold, torch.zeros_like(unit), unit)
            for index, sample in enumerate(batch):
                candidate_features = coarse_and_fine_features(sample, synsets, text_by_word, args.coarse)
                original_logits = scores(unit[index], candidate_features, model)
                if not original_logits[1] > original_logits[0]:
                    continue
                clipped_logits = scores(altered[index], candidate_features, model)
                if clipped_logits[1] > clipped_logits[0]:
                    newly_correct_images += 1
                original_margin = float((original_logits[1] - original_logits[0]).item())
                clipped_margin = float((clipped_logits[1] - clipped_logits[0]).item())
                original_margins.append(original_margin)
                clipped_margins.append(clipped_margin)
                margin_changes.append(clipped_margin - original_margin)
                original_coarse_logits.append(float(original_logits[0].item()))
                original_fine_logits.append(float(original_logits[1].item()))
                clipped_coarse_logits.append(float(clipped_logits[0].item()))
                clipped_fine_logits.append(float(clipped_logits[1].item()))
    print("Saving logit-margin changes and figures...", flush=True)
    np.savez_compressed(args.output_dir / "logit_margin_changes.npz", original_margin=original_margins, clipped_margin=clipped_margins, delta_margin=margin_changes, original_l_coarse=original_coarse_logits, original_l_fine=original_fine_logits, clipped_l_coarse=clipped_coarse_logits, clipped_l_fine=clipped_fine_logits)
    metrics = render_margin_histogram(args.output_dir, margin_changes, f"normalized threshold={threshold:.8g}; coarse={args.coarse}")
    render_logit_scatter(args.output_dir, original_coarse_logits, original_fine_logits, clipped_coarse_logits, clipped_fine_logits, f"normalized threshold={threshold:.8g}; coarse={args.coarse}")
    metrics.update({"threshold": threshold, "percentile": args.percentile, "feature_before_masking": "normalized", "operation": "set values with absolute magnitude below threshold to zero", "scoring_after_masking": "masked-normalized-feature", "coarse_label": args.coarse, "new_accuracy": newly_correct_images / len(margin_changes) if margin_changes else float("nan")})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wordnet-dir", type=Path, default=Path("~/wordnet").expanduser())
    parser.add_argument("--val-dir", type=Path, default=Path("data/val"))
    parser.add_argument("--clip-dir", type=Path, default=Path(__file__).resolve().parent / "CLIP")
    parser.add_argument("--output-dir", type=Path, default=Path("verifications/moderate_value_semantic"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--coarse", choices=("root", "middle"), default="middle", help="Use y1 (root) or the original middle node as y_coarse.")
    parser.add_argument("--percentile", type=float, default=99.0, help="Percentile of absolute normalized feature values used as the clipping threshold.")
    parser.add_argument("--max-images", type=int, help="Optional smoke-test limit; omit for all validation images.")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.text_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not 0.0 <= args.percentile <= 100.0:
        raise ValueError("--percentile must be between 0 and 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading and validating WordNet noun files...", flush=True)
    synsets, root = load_wordnet(args.wordnet_dir)
    samples = discover_samples(args.val_dir, synsets, root, args.max_images)
    print(f"Found {len(samples)} validation images. Loading CLIP RN50 on {args.device}...", flush=True)
    model, preprocess, clip_module = load_clip(args.clip_dir, args.device)
    text_by_word = encode_texts(model, clip_module, synsets, samples, args)
    threshold, accuracy = calculate_threshold(model, preprocess, text_by_word, synsets, samples, args)
    metrics = evaluate(threshold, model, preprocess, text_by_word, synsets, samples, args)
    report = {"configuration": {"clip_model": "RN50", "device": args.device, "images": len(samples), "root_excluded": root, "node_text": "first data.noun lemma verbatim", "coarse_label": args.coarse, "candidate_set": "{y_coarse, y_fine}"}, "original_two_text_accuracy": accuracy, "normalized_feature_threshold": metrics}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved complete report to {args.output_dir / 'summary.json'}", flush=True)
    print(f"normalized: threshold={metrics['threshold']:.8g}, E[delta_M]={metrics['mean_delta_margin']:.8g}, P(delta_M<0)={metrics['P_delta_margin_lt_0']:.6%}, correct={metrics['correct_images']}")


if __name__ == "__main__":
    main()
