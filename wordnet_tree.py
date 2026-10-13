"""Build a compact, reusable WordNet hierarchy for ImageNet-1K.

- Use noun WordNe.
- Map every ImageNet-1K class/WNID such as n01440764 to its WordNet noun synset.
- For each class, trace hypernyms toward the noun root. 
  At every multiple-parent choice, deterministically select the first parent, producing one unique path:
  y1 → y2 → … → yn, where yn is the ImageNet class and y1 is directly below the common root.
- Choose N no larger than the shortest path length
- Partition each path into N ordered bins and use each bin's last node as its representative.
  Treat the root as level 0, giving an (N+1)-level simplified tree.
  So the N+1 level = {1000 most detailed classes}
- Return, for every level j = 0…N, all unique representative nodes at that level.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WNID_RE = re.compile(r"n\d{8}")


class WordNetNouns:
    def __init__(self, root: Path):
        self.root = root
        self.offsets: dict[str, int] = {}
        self.synsets: dict[int, tuple[str, list[int]]] = {}
        self._read_index()
        self._read_data()

    def _read_index(self) -> None:
        for line in (self.root / "index.noun").read_text().splitlines():
            if not line or line.startswith(" "):
                continue
            fields = line.split()
            if len(fields) >= 6 and fields[0].isalpha():
                pointer_count = int(fields[3])
                self.offsets[fields[0]] = int(fields[6 + pointer_count])

    def _read_data(self) -> None:
        for line in (self.root / "data.noun").read_text().splitlines():
            if not line or line.startswith(" "):
                continue
            fields = line.split(" | ", 1)
            body = fields[0].split()
            offset = int(body[0])
            lemma_count = int(body[3], 16)
            lemmas = [body[4 + 2 * i].replace("_", " ") for i in range(lemma_count)]
            cursor = 4 + 2 * lemma_count
            parent_offsets = []
            for _ in range(int(body[cursor])):
                pointer = body[cursor + 1]
                target = int(body[cursor + 2])
                if pointer in {"@", "@i"}:
                    parent_offsets.append(target)
                cursor += 4
            self.synsets[offset] = (lemmas[0], parent_offsets)

    def path(self, wnid: str) -> list[str]:
        # ImageNet WNIDs encode the noun synset offset directly (nXXXXXXXX).
        offset = int(wnid[1:])
        if offset not in self.synsets:
            raise KeyError(f"ImageNet class {wnid} has no noun synset in {self.root}")
        result = []
        seen = set()
        while offset not in seen:
            seen.add(offset)
            name, parents = self.synsets[offset]
            result.append(name)
            if not parents:
                break
            offset = parents[0]
        result.reverse()
        return result


def class_ids(args: argparse.Namespace) -> list[str]:
    ids = sorted(p.name for p in Path(args.imagenet_root).iterdir() if p.is_dir())
    ids = [x for x in ids if WNID_RE.fullmatch(x)]
    if not ids:
        raise ValueError("No ImageNet WNIDs found; provide --classes or --imagenet-root")
    return ids


def build(args: argparse.Namespace) -> dict:
    wn = WordNetNouns(Path(args.wordnet))
    ids = class_ids(args)
    paths = {wnid: wn.path(wnid) for wnid in ids}
    min_depth = min(len(path) - 1 for path in paths.values())
    bins = args.bins if args.bins is not None else min_depth
    if bins < 1 or bins > min_depth:
        raise ValueError(f"--bins must be in [1, {min_depth}]")

    levels = {str(i): set() for i in range(bins + 1)}
    classes = {}
    for wnid, path in paths.items():
        reps = [path[0]]
        levels["0"].add(path[0])
        leaf_path = path[1:]
        for i in range(1, bins + 1):
            end = (i * len(leaf_path) + bins - 1) // bins
            representative = leaf_path[end - 1]
            reps.append(representative)
            levels[str(i)].add(representative)
        classes[wnid] = {"text": path[-1], "path": path, "representatives": reps}

    return {
        "num_classes": len(ids),
        "num_bins": bins,
        "parent_rule": "first noun hypernym in WordNet file order",
        "classes": classes,
        "levels": {key: sorted(value) for key, value in levels.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordnet", required=True, type=Path)
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.imagenet_root:
        parser.error("--imagenet-root is required")
    args.output.write_text(json.dumps(build(args), indent=2) + "\n")
    print(f"wrote {args.output} ({args.bins or 'automatic'} bins)")


if __name__ == "__main__":
    main()
