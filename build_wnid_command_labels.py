#!/usr/bin/env python3
import argparse
import json
import os
from nltk.corpus import wordnet as wn


def wnid_to_synset(wnid):
    pos = wnid[0]
    offset = int(wnid[1:])
    return wn.synset_from_pos_and_offset(pos, offset)


def synset_label(synset):
    return synset.lemma_names()[0].replace("_", " ")


def path_to_command_labels(path, skip_roots=True, start_at=None):
    labels = [synset_label(s) for s in path]

    if skip_roots:
        drop = {
            "entity",
            "physical entity",
            "object",
            "whole",
            "unit",
            "artifact",
            "instrumentality",
        }
        labels = [x for x in labels if x not in drop]

    if start_at is not None:
        if start_at not in labels:
            return None
        labels = labels[labels.index(start_at):]

    return labels


def collect_wnids(root_dir):
    wnids = []
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and len(name) == 9 and name[0].isalpha() and name[1:].isdigit():
            wnids.append(name)
    return wnids


def main():
    parser = argparse.ArgumentParser(
        description="Build command-label paths for all WNID folders under an ImageNet-style directory."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Directory containing WNID subfolders, e.g. data/imagenet/auxiliary_2000",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Optional JSONL output path, one WNID per line.",
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help='Optional coarse label to start from, e.g. "container".',
    )
    parser.add_argument(
        "--keep-roots",
        action="store_true",
        help="Keep very coarse WordNet roots such as entity / physical entity.",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    wnids = collect_wnids(root_dir)

    results = {}
    errors = {}

    for wnid in wnids:
        try:
            synset = wnid_to_synset(wnid)
            command_label_paths = []

            for path in synset.hypernym_paths():
                labels = path_to_command_labels(
                    path,
                    skip_roots=not args.keep_roots,
                    start_at=args.start_at,
                )
                if not labels:
                    continue

                command_label_paths.append({
                    "labels": labels,
                    "command": " ".join(f'"{label}"' for label in labels),
                })

            # 去重：有些 synset 的多条路径可能转成相同 labels
            deduped = []
            seen = set()
            for item in command_label_paths:
                key = tuple(item["labels"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)

            results[wnid] = {
                "wnid": wnid,
                "synset": synset.name(),
                "definition": synset.definition(),
                "lemma_names": [x.replace("_", " ") for x in synset.lemma_names()],
                "num_paths": len(deduped),
                "paths": deduped,
            }

        except Exception as exc:
            errors[wnid] = repr(exc)

    output = {
        "root_dir": root_dir,
        "num_wnids": len(wnids),
        "num_success": len(results),
        "num_errors": len(errors),
        "results": results,
        "errors": errors,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if args.output_jsonl is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for wnid in sorted(results):
                f.write(json.dumps(results[wnid], ensure_ascii=False) + "\n")

    print("Root dir:", root_dir)
    print("WNIDs:", len(wnids))
    print("Success:", len(results))
    print("Errors:", len(errors))
    print("Saved JSON:", args.output_json)
    if args.output_jsonl is not None:
        print("Saved JSONL:", args.output_jsonl)

    if errors:
        print("Some WNIDs failed:")
        for wnid, err in list(errors.items())[:20]:
            print(wnid, err)


if __name__ == "__main__":
    main()