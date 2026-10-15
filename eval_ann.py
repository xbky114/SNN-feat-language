"""Zero-shot ImageNet evaluation with reusable logits and metric artifacts."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import torch
from dataloader import imagenet_dataloader

def class_text_features(model, class_names, device):
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
    with torch.no_grad():
        text = model.encode_text(model.tokenize(prompts).to(device))
    return text / text.norm(dim=-1, keepdim=True)

@torch.no_grad()
def evaluate(model, loader, tree, device="cpu", print_interval=100, log=None):
    model.eval().to(device)
    bins = tree["num_bins"]
    level_keys = [str(k) for k in range(1, bins + 1)]
    levels = tree["levels"]
    text = {k: class_text_features(model, levels[k]["texts"], device)
            for k in ["0"] + level_keys}
    index = {k: levels[k]["class_to_idx"] for k in levels}
    path_indices = torch.tensor([
        [index[str(k)][tree["classes"][wnid]["representatives"][k]]
         for k in range(bins + 1)] for wnid in loader.dataset.classes], dtype=torch.long)
    correct = {k: 0 for k in level_keys}
    selected = {k: 0 for k in level_keys}
    depth_sum = total = 0
    full_logits, path_logits, saved_labels = [], [], []
    scale = model.logit_scale.exp()
    for batch_no, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        logits = {k: scale * features @ text[k].T for k in ["0"] + level_keys}
        targets = path_indices[labels.cpu()]
        path = torch.stack([logits[str(k)].gather(1, targets[:, k:k + 1].to(device)).squeeze(1)
                            for k in range(bins + 1)], dim=1)
        depth_sum += (path.softmax(1) * (torch.arange(bins + 1, device=device) / bins)).sum().item()
        for k in level_keys:
            correct[k] += (logits[k].argmax(1) == targets[:, int(k)].to(device)).sum().item()
            selected[k] += (path.argmax(1) == int(k)).sum().item()
        total += labels.numel()
        full_logits.append(torch.cat([logits[k].cpu() for k in ["0"] + level_keys], 1))
        path_logits.append(path.cpu()); saved_labels.append(labels.cpu())
        if log and batch_no % print_interval == 0:
            print(f"batch={batch_no} samples={total} semantic_depth={depth_sum / total:.6f} "
                  f"accuracy={json.dumps({k: correct[k] / total for k in level_keys})} "
                  f"selectivity={json.dumps({k: selected[k] / total for k in level_keys})}",
                  file=log, flush=True)
    summary = {"samples": total, "semantic_depth": depth_sum / total,
               "accuracy": {k: correct[k] / total for k in level_keys},
               "selectivity": {k: selected[k] / total for k in level_keys}}
    # Split the concatenated level logits into independently addressable tensors.
    widths = [levels[k]["num_classes"] for k in ["0"] + level_keys]
    merged = torch.cat(full_logits)
    artifact = {"path_logits": torch.cat(path_logits)}
    start = widths[0]  # level 0 has one class and is retained only in path_logits
    for key, width in zip(level_keys, widths[1:]):
        artifact[f"logits_{key}"] = merged[:, start:start + width]
        start += width
    return summary, artifact

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--clip", required=True)
    p.add_argument("--tree", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--print-interval", type=int, default=20)
    a = p.parse_args()
    from CLIPs import get_clip
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "evaluation.log").open("w") as log, contextlib.redirect_stdout(log):
        print(f"arguments={json.dumps(vars(a), sort_keys=True, default=str)}", file=log, flush=True)
        adapter = get_clip(a.clip, device=a.device)
        tree = json.loads(Path(a.tree).read_text())
        evaluation = imagenet_dataloader(a.data, batch_size=a.batch_size,
                                         num_workers=a.workers, image_size=adapter.input_resolution)
        summary, artifact = evaluate(adapter, evaluation, tree, a.device, a.print_interval, log)
        torch.save(artifact, out / "logits.pt")
        (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, sort_keys=True), file=log, flush=True)

if __name__ == "__main__":
    main()
