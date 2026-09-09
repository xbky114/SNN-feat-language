#!/usr/bin/env python3
"""Test q = -gamma c + r for CLIP RN50 on ImageNet validation images.

``h`` is the pre-activation at RN50's last residual addition.  Here e1 is the
embedding of the requested middle-path ``y_coarse`` label.  WordNet lemmas are
passed to CLIP verbatim.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from wordnet_geometry import Synset, leaf_paths, parse_data_noun, parse_index_noun, parse_noun_senses, validate_indexes


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Item:
    image: Path
    leaf: str
    path: tuple[str, ...]


def load_wordnet(directory: Path) -> tuple[dict[str, Synset], str]:
    files = {name: directory / name for name in ("data.noun", "index.noun", "index.sense")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing WordNet files: " + ", ".join(missing))
    synsets = parse_data_noun(files["data.noun"])
    index = parse_index_noun(files["index.noun"])
    senses = parse_noun_senses(files["index.sense"])
    validate_indexes(synsets, index, senses)
    root, _ = leaf_paths(synsets)
    return synsets, root


def path_to_root(leaf: str, synsets: dict[str, Synset], root: str) -> tuple[str, ...]:
    reverse, seen = [], set()
    node: str | None = leaf
    while node is not None:
        if node in seen:
            raise ValueError(f"Cycle on selected hypernym path from {leaf}")
        if node not in synsets:
            raise KeyError(f"ImageNet class {leaf} is absent from data.noun")
        seen.add(node)
        reverse.append(node)
        node = synsets[node].parent
    path = tuple(reversed(reverse))
    if not path or path[0] != root or len(path) < 2:
        raise ValueError(f"Class {leaf} does not have a path below shared root {root}")
    return path[1:]


class ImageNetPaths(Dataset):
    def __init__(self, val_dir: Path, preprocess, synsets: dict[str, Synset], root: str, max_images: int | None):
        self.preprocess = preprocess
        self.items: list[Item] = []
        for class_dir in sorted(path for path in val_dir.iterdir() if path.is_dir()):
            name = class_dir.name
            leaf = name[1:] if name.startswith("n") and len(name) == 9 and name[1:].isdigit() else name
            selected = path_to_root(leaf, synsets, root)
            for image in sorted(class_dir.iterdir()):
                if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                    self.items.append(Item(image, leaf, selected))
                    if max_images is not None and len(self.items) >= max_images:
                        return
        if not self.items:
            raise ValueError(f"No images found under {val_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(item.image) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return tensor, index


def exact_float16_percentile(histogram: np.ndarray, percentile: float) -> float:
    """Percentile of all finite fp16 activations represented by a bit histogram."""
    bit_patterns = np.arange(65536, dtype=np.uint16)
    values = bit_patterns.view(np.float16)
    finite = np.isfinite(values) & (histogram > 0)
    values, counts = values[finite], histogram[finite]
    order = np.argsort(values.astype(np.float32), kind="stable")
    values, cumulative = values[order].astype(np.float64), np.cumsum(counts[order], dtype=np.int64)
    total = int(cumulative[-1])
    rank = (total - 1) * percentile / 100.0

    def at(k: int) -> float:
        return float(values[np.searchsorted(cumulative, k + 1, side="left")])

    lower, upper = int(np.floor(rank)), int(np.ceil(rank))
    fraction = rank - lower
    return at(lower) + fraction * (at(upper) - at(lower))


def collate(batch):
    import torch
    images, indices = zip(*batch)
    return torch.stack(images), torch.tensor(indices, dtype=torch.long)


def make_loader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                      pin_memory=True, persistent_workers=args.num_workers > 0, collate_fn=collate)


def encode_path_texts(model, clip, dataset: ImageNetPaths, synsets: dict[str, Synset], args):
    import torch
    offsets = sorted({offset for item in dataset.items for offset in item.path})
    # First lemma exactly as data.noun stores it: no prompt, article, or underscore replacement.
    words = [synsets[offset].words[0] for offset in offsets]
    chunks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(words), args.text_batch_size), desc="Encoding WordNet text"):
            tokens = clip.tokenize(words[start:start + args.text_batch_size]).to(args.device)
            feature = model.encode_text(tokens).float()
            chunks.append(feature / feature.norm(dim=-1, keepdim=True))
    return {offset: feature for offset, feature in zip(offsets, torch.cat(chunks))}


def plot_histogram(values: np.ndarray, xlabel: str, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.hist(values, bins="auto", color="#4477aa", edgecolor="white")
    axis.set(xlabel=xlabel, ylabel="valid images")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("./data/auxiliary_2000"))
    parser.add_argument("--wordnet-dir", type=Path, default=Path("~/wordnet").expanduser())
    parser.add_argument("--clip-dir", type=Path, default=Path(__file__).resolve().parent / "CLIP")
    parser.add_argument("--output-dir", type=Path, default=Path("./verifications/v_f_infinity_e1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--max-images", type=int, default=None, help="Development/testing subset; default uses all images")
    return parser.parse_args()


def main():
    import torch
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("This experiment requires --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    sys.path.insert(0, str(args.clip_dir))
    import clip

    synsets, root = load_wordnet(args.wordnet_dir)
    model, preprocess = clip.load("RN50", device=args.device, jit=False)
    model.eval()
    visual = model.visual
    if not hasattr(visual, "attnpool"):
        raise TypeError("RN50 ResNet visual tower with AttentionPool2d is required")
    dataset = ImageNetPaths(args.data, preprocess, synsets, root, args.max_images)
    loader = make_loader(dataset, args)
    text = encode_path_texts(model, clip, dataset, synsets, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    captured: list[torch.Tensor] = []
    def hook(_module, inputs):
        captured.append(inputs[0].detach().clone())
    handle = visual.layer4[-1].relu3.register_forward_pre_hook(hook)
    histogram = np.zeros(65536, dtype=np.int64)
    try:
        with torch.inference_mode():
            for images, _ in tqdm(loader, desc="Threshold pass", total=len(loader)):
                captured.clear()
                model.encode_image(images.to(args.device, non_blocking=True))
                if len(captured) != 1:
                    raise RuntimeError(f"Expected one h capture, got {len(captured)}")
                array = captured[0].cpu().numpy()
                if array.dtype != np.float16:
                    array = array.astype(np.float16)
                histogram += np.bincount(array.reshape(-1).view(np.uint16), minlength=65536)
        vth = exact_float16_percentile(histogram, args.percentile)

        gammas, rhos, winners, coarse_scores, fine_scores = [], [], [], [], []
        eps = torch.finfo(torch.float32).eps
        for images, indices in tqdm(loader, desc="Geometry pass", total=len(loader)):
            captured.clear()
            with torch.no_grad():
                model.encode_image(images.to(args.device, non_blocking=True))
                h = captured[0]
            h_inf = torch.clamp(h, min=0.0, max=vth).detach().requires_grad_(True)
            v = ((h > 0) & (h < vth)).to(h.dtype)

            def G(value):
                return visual.attnpool(value).float()
            f_inf, u = torch.autograd.functional.jvp(G, (h_inf,), (v,), create_graph=False, strict=True)
            z = f_inf / f_inf.norm(dim=-1, keepdim=True)
            q = u - z * (u * z).sum(dim=-1, keepdim=True)

            for row, data_index in enumerate(indices.tolist()):
                item = dataset.items[data_index]
                # Zero-based n//2 implements: odd -> middle; even -> the
                # later (+1 in one-based wording) of the two middle nodes.
                coarse_index = len(item.path) // 2
                e1 = text[item.path[coarse_index]].float()
                c = e1 - z[row] * torch.dot(z[row], e1)
                q_norm, c_sq = q[row].norm(), torch.dot(c, c)
                # Record the requested path-restricted current match and coarse/fine scores.
                path_features = torch.stack([text[offset] for offset in item.path]).float()
                scores = path_features @ z[row]
                winners.append(int(torch.argmax(scores).item()))
                coarse_scores.append(float(scores[len(item.path) // 2].item()))
                fine_scores.append(float(scores[-1].item()))
                if q_norm > eps and c_sq > eps:
                    gamma = -torch.dot(q[row], c) / c_sq
                    residual = q[row] + gamma * c
                    gammas.append(float(gamma.item()))
                    rhos.append(float((residual.norm() / q_norm).item()))
    finally:
        handle.remove()

    gamma_array, rho_array = np.asarray(gammas), np.asarray(rhos)
    if not len(gamma_array):
        raise RuntimeError("No sample had both nonzero c and nonzero q")
    np.savez_compressed(args.output_dir / "per_image_results.npz", gamma=gamma_array, rho=rho_array,
                        winner_path_index=np.asarray(winners), coarse_score=np.asarray(coarse_scores),
                        fine_score=np.asarray(fine_scores))
    plot_histogram(gamma_array, r"$\hat{\gamma}$", args.output_dir / "gamma_histogram.png")
    plot_histogram(rho_array, r"$\rho$", args.output_dir / "rho_histogram.png")
    summary = {
        "configuration": {"model": "RN50", "device": args.device, "data": str(args.data),
                          "wordnet_dir": str(args.wordnet_dir), "percentile": args.percentile,
                          "num_images": len(dataset), "max_images": args.max_images,
                          "h_definition": "input to final layer4 relu3 (after residual addition)",
                          "v_definition": "strict 0 < h < V_th",
                          "e1_definition": "normalized CLIP embedding of y_coarse (path index len(path)//2)"},
        "V_th": vth,
        "total_h_elements": int(histogram.sum()),
        "valid_samples": int(len(gamma_array)),
        "gamma_positive_proportion": float((gamma_array > 0).mean()),
        "rho_mean": float(rho_array.mean()),
        "rho_median": float(np.median(rho_array)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
