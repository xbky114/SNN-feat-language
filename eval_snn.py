"""Evaluate an SNN visual tower with an ANN CLIP text tower.

The evaluator keeps logits for every timestep and writes image-based shards so
the complete temporal record can be analysed without one very large file.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from CLIPs import get_clip
from clip_with_snn_visual import CLIPWithSNNVisual
from dataloader import imagenet_dataloader
from to_snn.prototype import SNNArtifact


def _coding(name: str):
    fn = getattr(importlib.import_module("to_snn.coding"), name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"unknown coding function: {name!r}")
    return fn


def _text_features(model, names, device):
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in names]
    tokens = model.tokenize(prompts).to(device)
    with torch.no_grad():
        out = model.encode_text(tokens)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _metric_line(batch, total, depth, correct, selected, levels):
    return (f"batch={batch} samples={total} semantic_depth={depth / total:.6f} "
            f"accuracy={json.dumps({k: correct[k] / total for k in levels})} "
            f"selectivity={json.dumps({k: selected[k] / total for k in levels})}")


@torch.no_grad()
def evaluate(model, loader, tree, device, timesteps, images_per_file,
             output_dir, print_interval, log=None):
    model.eval().to(device)
    bins = int(tree["num_bins"])
    levels = [str(k) for k in range(1, bins + 1)]
    level_names = tree["levels"]
    indices = {k: level_names[k]["class_to_idx"] for k in level_names}
    paths = torch.tensor([
        [indices[str(k)][tree["classes"][wnid]["representatives"][k]]
         for k in range(bins + 1)]
        for wnid in loader.dataset.classes], dtype=torch.long)

    total = 0
    depth = torch.zeros(timesteps, dtype=torch.float64)
    correct = {k: torch.zeros(timesteps, dtype=torch.float64) for k in levels}
    selected = {k: torch.zeros(timesteps, dtype=torch.float64) for k in levels}
    shard_path, shard_levels, shard_count, shard_no = [], {k: [] for k in levels}, 0, 0

    def flush_shard():
        nonlocal shard_path, shard_levels, shard_count, shard_no
        if not shard_count:
            return
        artifact = {"path_logits": torch.cat(shard_path, dim=1)}
        for key in levels:
            artifact[f"logits_{key}"] = torch.cat(shard_levels[key], dim=1)
        torch.save(artifact, output_dir / f"logits_{shard_no:05d}.pt")
        shard_no += 1
        shard_path, shard_levels, shard_count = [], {k: [] for k in levels}, 0

    for batch_no, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        # One call per hierarchy level; each result is [T, batch, classes].
        logits = {k: model.calc_logits_through_time(
            images, [f"a photo of a {name.replace('_', ' ')}"
                     for name in level_names[k]["texts"]], timesteps)
            for k in ["0"] + levels}
        targets = paths[labels.cpu()]
        path = torch.stack([
            logits[str(k)].gather(2, targets[:, k].to(device)[None, :, None]
                                   .expand(timesteps, -1, 1)).squeeze(-1)
            for k in range(bins + 1)], dim=2)
        depth += (path.softmax(2).cpu() *
                  (torch.arange(bins + 1, dtype=torch.float64) / bins)).sum((1, 2))
        for k in levels:
            pred = logits[k].argmax(2).cpu()
            correct[k] += (pred == targets[:, int(k)][None, :]).sum(1)
            selected[k] += (path.argmax(2).cpu() == int(k)).sum(1)
        n = labels.numel(); total += n
        # Split a batch when necessary so each shard has at most the
        # configured number of images (the timestep axis is never sharded).
        start = 0
        while start < n:
            take = min(images_per_file - shard_count, n - start)
            shard_path.append(path[:, start:start + take].cpu())
            for k in levels:
                shard_levels[k].append(logits[k][:, start:start + take].cpu())
            shard_count += take
            start += take
            if shard_count == images_per_file:
                flush_shard()
        if batch_no % print_interval == 0:
            line = _metric_line(batch_no, total, depth[-1].item(),
                                {k: correct[k][-1].item() for k in levels},
                                {k: selected[k][-1].item() for k in levels}, levels)
            print(line, file=log, flush=True) if log is not None else print(line, flush=True)
    flush_shard()
    per_time = [{"timestep": t + 1, "semantic_depth": depth[t].item() / total,
                 "accuracy": {k: correct[k][t].item() / total for k in levels},
                 "selectivity": {k: selected[k][t].item() / total for k in levels}}
                for t in range(timesteps)]
    return {"samples": total, "timesteps": timesteps,
            "final": per_time[-1], "per_timestep": per_time}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--clip", required=True)
    p.add_argument("--snn", required=True); p.add_argument("--tree", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--coding", default="rate_coding")
    p.add_argument("--T", type=int, default=256); p.add_argument("--images-per-file", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--print-interval", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.T <= 0 or a.images_per_file <= 0 or a.print_interval <= 0:
        p.error("--T, --images-per-file, and --print-interval must be positive")
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    log_path = out / "evaluation.log"
    with log_path.open("w") as log:
        print(f"arguments={json.dumps(vars(a), sort_keys=True)}", file=log, flush=True)
        ann = get_clip(a.clip, device=a.device)
        snn = SNNArtifact.load(a.snn, map_location=a.device)
        model = CLIPWithSNNVisual(ann, snn, _coding(a.coding))
        tree = json.loads(Path(a.tree).read_text())
        loader = imagenet_dataloader(a.data, batch_size=a.batch_size,
                                     num_workers=a.workers, image_size=model.input_resolution)
        summary = evaluate(model, loader, tree, a.device, a.T,
                           a.images_per_file, out, a.print_interval, log)
        print(f"final_summary={json.dumps(summary['final'], sort_keys=True)}", file=log, flush=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
