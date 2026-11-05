"""Evaluate ANN CLIP on the fixed three-level hierarchy."""
import argparse
import json
from pathlib import Path

import torch

from CLIPs import get_clip
from dataloader import DATASETS, build_dataloader
from prompt import build_prompt


@torch.no_grad()
def evaluate(model, loader, device="cpu", prompt_father=False):
    dataset = loader.dataset
    names = dataset.class_names
    levels = list(names)
    index = {level: {name: i for i, name in enumerate(names[level])} for level in levels}
    prompts = {
        "basic": [build_prompt(name) for name in names["basic"]],
        "subordinate": [build_prompt(name, dataset.parent["subordinate"][name]) if prompt_father else build_prompt(name) for name in names["subordinate"]],
        "fine_grained": [build_prompt(name, dataset.parent["fine_grained"][name]) if prompt_father else build_prompt(name) for name in names["fine_grained"]],
    }
    text = {level: model.encode_text(model.tokenize(prompts[level]).to(device)) for level in levels}
    text = {level: value / value.norm(dim=-1, keepdim=True) for level, value in text.items()}
    correct = {level: 0 for level in levels}
    selected = {level: 0 for level in levels}
    depth = 0.0
    total = 0
    saved = {level: [] for level in levels}
    paths = []
    for batch in loader:
        images = batch["image"].to(device)
        batch_size = images.shape[0]
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        logits = {level: model.logit_scale.exp() * features @ text[level].T for level in levels}
        targets = {level: torch.tensor([index[level][value] for value in batch[level]], device=device) for level in levels}
        path = torch.stack([logits[level][torch.arange(batch_size, device=device), targets[level]] for level in levels], dim=1)
        probabilities = path.softmax(dim=1)
        paths.append(path.cpu())
        for level in levels:
            saved[level].append(logits[level].cpu())
        depth += (probabilities * torch.tensor([0.0, 0.5, 1.0], device=device)).sum(dim=1).sum().item()
        for i, level in enumerate(levels):
            correct[level] += (logits[level].argmax(dim=1) == targets[level]).sum().item()
            selected[level] += (path.argmax(dim=1) == i).sum().item()
        total += batch_size
    return {
        "samples": total,
        "semantic_depth": depth / total,
        "accuracy": {level: correct[level] / total for level in levels},
        "selectivity": {level: selected[level] / total for level in levels},
        "_logits": {level: torch.cat(saved[level]) for level in levels},
        "_path_logits": torch.cat(paths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt-father", action="store_true")
    args = parser.parse_args()
    model = get_clip(args.clip, device=args.device).eval()
    loader = build_dataloader(args.dataset, args.data, batch_size=args.batch_size, image_size=model.input_resolution, num_workers=args.workers)
    summary = evaluate(model, loader, args.device, args.prompt_father)
    logits = summary.pop("_logits")
    path_logits = summary.pop("_path_logits")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({**{f"logits_{level}": value for level, value in logits.items()}, "path_logits": path_logits}, output_dir / "logits.pt")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
