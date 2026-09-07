#!/usr/bin/env python3
"""Measure geometric properties of CLIP embeddings along WordNet noun paths."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Synset:
    offset: str
    words: tuple[str, ...]
    hypernyms: tuple[str, ...]

    @property
    def parent(self) -> str | None:
        return self.hypernyms[0] if self.hypernyms else None


def parse_data_noun(path: Path) -> dict[str, Synset]:
    """Parse noun synsets, choosing the first @/@i pointer as the parent."""
    synsets: dict[str, Synset] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split(" | ", 1)[0].split()
            if not fields or len(fields[0]) != 8 or not fields[0].isdigit():
                continue

            try:
                offset = fields[0]
                word_count = int(fields[3], 16)
                words = tuple(fields[4 : 4 + 2 * word_count : 2])
                cursor = 4 + 2 * word_count
                pointer_count = int(fields[cursor])
                cursor += 1
                hypernyms = []
                for _ in range(pointer_count):
                    symbol, target, pos, _source_target = fields[cursor : cursor + 4]
                    cursor += 4
                    if pos == "n" and symbol in {"@", "@i"}:
                        hypernyms.append(target)
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Malformed {path}:{line_number}") from exc

            if not words:
                raise ValueError(f"Synset {offset} in {path} has no words")
            synsets[offset] = Synset(offset, words, tuple(hypernyms))

    if not synsets:
        raise ValueError(f"No noun synsets found in {path}")
    return synsets


def parse_index_noun(path: Path) -> dict[str, tuple[str, ...]]:
    entries: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) < 6 or fields[1] != "n":
                continue
            try:
                synset_count = int(fields[2])
                pointer_count = int(fields[3])
                offset_start = 6 + pointer_count
                offsets = tuple(fields[offset_start : offset_start + synset_count])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Malformed {path}:{line_number}") from exc
            if len(offsets) != synset_count:
                raise ValueError(f"Malformed {path}:{line_number}: missing offsets")
            entries[fields[0]] = offsets
    return entries


def parse_noun_senses(path: Path) -> list[tuple[str, str]]:
    senses: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"Malformed {path}:{line_number}")
            sense_key = fields[0]
            percent = sense_key.rfind("%")
            if percent >= 0 and sense_key[percent + 1 : percent + 2] == "1":
                senses.append((sense_key[:percent], fields[1]))
    return senses


def validate_indexes(
    synsets: dict[str, Synset],
    noun_index: dict[str, tuple[str, ...]],
    noun_senses: Iterable[tuple[str, str]],
) -> None:
    for lemma, offsets in noun_index.items():
        for offset in offsets:
            if offset not in synsets:
                raise ValueError(f"index.noun references missing noun synset {offset}")
            if lemma.lower() not in {word.lower() for word in synsets[offset].words}:
                raise ValueError(f"index.noun lemma {lemma!r} is absent from synset {offset}")
    for lemma, offset in noun_senses:
        if offset not in synsets:
            raise ValueError(f"index.sense references missing noun synset {offset}")
        if lemma.lower() not in {word.lower() for word in synsets[offset].words}:
            raise ValueError(f"index.sense lemma {lemma!r} is absent from synset {offset}")


def leaf_paths(synsets: dict[str, Synset]) -> tuple[str, list[list[str]]]:
    roots = [offset for offset, synset in synsets.items() if synset.parent is None]
    if len(roots) != 1:
        raise ValueError(f"Expected one noun root, found {len(roots)}: {roots}")
    root = roots[0]

    has_child: set[str] = set()
    for synset in synsets.values():
        for hypernym in synset.hypernyms:
            if hypernym not in synsets:
                raise ValueError(f"Synset {synset.offset} has missing hypernym {hypernym}")
            has_child.add(hypernym)

    paths: list[list[str]] = []
    for leaf in sorted(set(synsets) - has_child):
        reverse_path: list[str] = []
        seen: set[str] = set()
        current: str | None = leaf
        while current is not None:
            if current in seen:
                raise ValueError(f"Cycle detected while following synset {leaf}")
            seen.add(current)
            reverse_path.append(current)
            current = synsets[current].parent
        path = list(reversed(reverse_path))
        if path[0] != root:
            raise ValueError(f"Path from leaf {leaf} does not reach root {root}")
        # The shared WordNet root itself is excluded: y1 is its child.
        paths.append(path[1:])
    return root, paths


def encode_words(words: list[str], batch_size: int, device: str, clip_dir: Path):
    import torch

    if device != "cuda:0":
        raise ValueError("This experiment is fixed to device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cuda:0 is required")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("cuda:0 does not exist")

    sys.path.insert(0, str(clip_dir))
    import clip  # type: ignore

    model, _ = clip.load("RN50", device=device, jit=False)
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(words), batch_size):
            tokens = clip.tokenize(words[start : start + batch_size]).to(device)
            embedding = model.encode_text(tokens).float()
            embedding /= embedding.norm(dim=-1, keepdim=True)
            batches.append(embedding.cpu())
    return torch.cat(batches, dim=0)


def directional_consistency(values, direction: str) -> float:
    """Fraction of adjacent steps whose strict direction matches ``direction``."""
    differences = values[1:] - values[:-1]
    if direction == "decreasing":
        return float((differences < 0).float().mean().item())
    if direction == "increasing":
        return float((differences > 0).float().mean().item())
    raise ValueError(f"Unsupported direction: {direction}")


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    required = {name: args.wordnet_dir / name for name in ("data.noun", "index.noun", "index.sense")}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing WordNet files: " + ", ".join(missing))

    synsets = parse_data_noun(required["data.noun"])
    noun_index = parse_index_noun(required["index.noun"])
    noun_senses = parse_noun_senses(required["index.sense"])
    validate_indexes(synsets, noun_index, noun_senses)
    shared_root, paths = leaf_paths(synsets)

    # A synset is represented by its first WordNet lemma, with no prompt or rewriting.
    texts = sorted({synsets[offset].words[0] for path in paths for offset in path})
    embeddings = encode_words(texts, args.batch_size, args.device, args.clip_dir)
    embedding_by_text = {text: embeddings[i] for i, text in enumerate(texts)}

    nonnegative_count = 0
    consistency_sums = [0.0, 0.0, 0.0]
    for path in paths:
        path_embeddings = torch.stack([embedding_by_text[synsets[offset].words[0]] for offset in path])
        e1, en = path_embeddings[0], path_embeddings[-1]
        root_similarity = path_embeddings @ e1
        leaf_similarity = path_embeddings @ en
        end_to_root = torch.dot(en, e1)
        q = root_similarity - leaf_similarity * end_to_root

        nonnegative_count += int(end_to_root.item() >= 0.0)
        consistency_sums[0] += directional_consistency(root_similarity, "decreasing")
        consistency_sums[1] += directional_consistency(leaf_similarity, "increasing")
        consistency_sums[2] += directional_consistency(q, "decreasing")

    total = len(paths)
    properties = {
        "e_n_dot_e_1_nonnegative": {
            "satisfied": nonnegative_count,
            "total": total,
            "proportion": nonnegative_count / total,
        },
        "e_1_dot_e_k_decreasing_direction_consistency": {
            "mean_over_paths": consistency_sums[0] / total,
            "direction": "decreasing",
        },
        "e_n_dot_e_k_increasing_direction_consistency": {
            "mean_over_paths": consistency_sums[1] / total,
            "direction": "increasing",
        },
        "q_k_decreasing_direction_consistency": {
            "mean_over_paths": consistency_sums[2] / total,
            "direction": "decreasing",
        },
    }
    return {
        "configuration": {
            "wordnet_dir": str(args.wordnet_dir),
            "clip_model": "RN50",
            "device": args.device,
            "batch_size": args.batch_size,
            "node_text": "first lemma exactly as stored in data.noun; no prompt or rewriting",
            "path_rule": "first @ or @i noun hypernym; shared top root excluded",
        },
        "dataset": {
            "noun_synsets": len(synsets),
            "noun_index_lemmas": len(noun_index),
            "noun_senses": len(noun_senses),
            "shared_top_root_offset": shared_root,
            "shared_top_root_text": synsets[shared_root].words[0],
            "leaf_paths": total,
            "encoded_unique_texts": len(texts),
        },
        "properties": properties,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wordnet-dir", type=Path, default=Path("~/wordnet").expanduser())
    parser.add_argument("--clip-dir", type=Path, default=Path(__file__).resolve().parent / "CLIP")
    parser.add_argument("--output", type=Path, default=Path("results/wordnet_geometry.json"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Evaluated {result['dataset']['leaf_paths']} leaf paths")
    nonnegative = result["properties"]["e_n_dot_e_1_nonnegative"]
    print(
        "e_n_dot_e_1_nonnegative: "
        f"{nonnegative['satisfied']}/{nonnegative['total']} "
        f"({nonnegative['proportion']:.6%})"
    )
    for name, measurement in result["properties"].items():
        if name == "e_n_dot_e_1_nonnegative":
            continue
        print(f"{name}: {measurement['mean_over_paths']:.6%}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
