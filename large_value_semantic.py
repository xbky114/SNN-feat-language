#!/usr/bin/env python3
"""Test whether clipping large RN50 CLIP image-feature values shifts predictions coarseward.

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
            texts = [synsets[offset].words[0] for offset in sample.path]
            text_features = [text_by_word[text] for text in texts]
            baseline = torch.softmax(scores(unit[index], text_features, model), dim=0)
            if int(baseline.argmax()) == len(sample.path) - 1:
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
        raise ValueError("No images were originally classified with yn as the highest-scoring path text")
    normalized_values.flush()
    print(f"Computing exact normalized-feature {args.percentile:g}th percentile...", flush=True)
    threshold = float(np.percentile(normalized_values[:position], args.percentile))
    # Persist exactly the retained values, without unused zero-filled capacity.
    np.save(args.output_dir / "absolute_normalized_image_features.npy", normalized_values[:position])
    del normalized_values
    (args.output_dir / ".absolute_normalized_image_features.tmp.npy").unlink()
    print(f"Threshold ready: normalized={threshold:.8g}", flush=True)
    accuracy = correct_images / len(samples)
    print(f"Original path-label accuracy: {correct_images}/{len(samples)} ({accuracy:.6%})", flush=True)
    return threshold, {"total_images": len(samples), "correct_images": correct_images, "accuracy": accuracy}


def encode_texts(model, clip_module, synsets: dict[str, Synset], samples: list[Sample], args: argparse.Namespace):
    import torch

    # First data.noun lemma verbatim: no article, template, or other prompt wrapper.
    texts = sorted({synsets[offset].words[0] for sample in samples for offset in sample.path})
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


def render(mode_dir: Path, coarse: list[float], fine: list[float], shifts: list[float], title: str) -> dict[str, float | int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coarse_array, fine_array, shift_array = map(np.asarray, (coarse, fine, shifts))
    mean = float(shift_array.mean()) if len(shift_array) else float("nan")
    probability = float((shift_array > 0).mean()) if len(shift_array) else float("nan")
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(fine_array, coarse_array, s=5, alpha=0.35, rasterized=True)
    limits = np.array([fine_array.min(), fine_array.max(), coarse_array.min(), coarse_array.max()]) if len(shifts) else np.array([-1, 1])
    low, high = float(limits.min()), float(limits.max())
    axis.plot([low, high], [low, high], "k--", linewidth=1, label="y=x")
    axis.set(xlabel=r"$\Delta P_{fine}$", ylabel=r"$\Delta P_{coarse}$", title=title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(mode_dir / "coarse_vs_fine_scatter.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.hist(shift_array, bins="auto", color="#4477aa", edgecolor="white")
    axis.axvline(0, color="black", linestyle="--", label="zero")
    axis.axvline(mean, color="#cc3311", label=f"mean = {mean:.6g}")
    axis.set(xlabel=r"$S=\Delta P_{coarse}-\Delta P_{fine}$", ylabel="images", title=title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(mode_dir / "semantic_shift_histogram.png", dpi=180)
    plt.close(fig)
    return {"correct_images": len(shifts), "mean_S_coarse": mean, "P_S_coarse_gt_0": probability}


def evaluate(threshold, model, preprocess, text_by_word, synsets, samples, args):
    import torch

    coarse_changes: list[float] = []
    fine_changes: list[float] = []
    shifts: list[float] = []
    print(f"Evaluating normalized-feature clipping (threshold={threshold:.8g}, coarse={args.coarse})...", flush=True)
    with torch.inference_mode():
        for raw, batch in image_features(model, preprocess, samples, args.device, args.batch_size, "Clipping evaluation"):
            unit = raw / raw.norm(dim=-1, keepdim=True)
            # The requested normalized-feature operation does not re-normalize after clipping.
            altered = torch.clamp(unit, min=-threshold, max=threshold)
            for index, sample in enumerate(batch):
                texts = [synsets[offset].words[0] for offset in sample.path]
                text_features = [text_by_word[text] for text in texts]
                baseline = torch.softmax(scores(unit[index], text_features, model), dim=0)
                if int(baseline.argmax()) != len(sample.path) - 1:
                    continue
                changed = torch.softmax(scores(altered[index], text_features, model), dim=0)
                coarse_index = 0 if args.coarse == "root" else len(sample.path) // 2
                coarse_delta = float((changed[coarse_index] - baseline[coarse_index]).item())
                fine_delta = float((changed[-1] - baseline[-1]).item())
                coarse_changes.append(coarse_delta)
                fine_changes.append(fine_delta)
                shifts.append(coarse_delta - fine_delta)
    print("Saving probability changes and figures...", flush=True)
    np.savez_compressed(args.output_dir / "probability_changes.npz", delta_p_coarse=coarse_changes, delta_p_fine=fine_changes, semantic_shift=shifts)
    metrics = render(args.output_dir, coarse_changes, fine_changes, shifts, f"normalized threshold={threshold:.8g}; coarse={args.coarse}")
    metrics.update({"threshold": threshold, "percentile": args.percentile, "feature_before_clipping": "normalized", "scoring_after_clipping": "clipped-normalized-feature", "coarse_label": args.coarse})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wordnet-dir", type=Path, default=Path("~/wordnet").expanduser())
    parser.add_argument("--val-dir", type=Path, default=Path("data/val"))
    parser.add_argument("--clip-dir", type=Path, default=Path(__file__).resolve().parent / "CLIP")
    parser.add_argument("--output-dir", type=Path, default=Path("verifications/large_value_semantic"))
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
    report = {"configuration": {"clip_model": "RN50", "device": args.device, "images": len(samples), "root_excluded": root, "node_text": "first data.noun lemma verbatim", "coarse_label": args.coarse}, "original_path_label_accuracy": accuracy, "normalized_feature_threshold": metrics}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved complete report to {args.output_dir / 'summary.json'}", flush=True)
    print(f"normalized: threshold={metrics['threshold']:.8g}, E[S_coarse]={metrics['mean_S_coarse']:.8g}, P(S_coarse>0)={metrics['P_S_coarse_gt_0']:.6%}, correct={metrics['correct_images']}")


if __name__ == "__main__":
    main()
