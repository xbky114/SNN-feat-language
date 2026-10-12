"""Evaluate a saved visual SNN artifact with the ANN CLIP text tower."""
from __future__ import annotations
import argparse
import torch
from dataloader import imagenet_dataloaders, resolve_class_names
from eval_ann import class_text_features
from to_snn.prototype import SNNArtifact
from CLIPs import get_clip

@torch.no_grad()
def evaluate_snn(model, pool, artifact, loader, class_names, timesteps=128, device="cpu"):
    model.eval().to(device); pool.eval().to(device); artifact.eval().to(device)
    text_features = class_text_features(model, class_names, device)
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        artifact.reset_state(); feature_sum = None
        for _ in range(timesteps):
            features = artifact(images)
            feature_sum = features if feature_sum is None else feature_sum + features
        # OpenAI CLIP leaves positional embeddings in fp32 while the
        # attention projection weights are fp16 on CUDA.
        pool_dtype = pool.q_proj.weight.dtype
        features = pool((feature_sum / timesteps).to(dtype=pool_dtype))
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logits = model.logit_scale.exp() * features @ text_features.T
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item(); total += labels.numel()
    return correct / total if total else 0.0

def main():
    p = argparse.ArgumentParser(); p.add_argument("--snn", required=True)
    p.add_argument("--data", required=True); p.add_argument("--timesteps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = p.parse_args()
    adapter = get_clip("openai_rn50", device=a.device)
    model, pool = adapter.model, adapter.pool
    artifact = SNNArtifact.load(a.snn, map_location="cpu")
    evaluation, _ = imagenet_dataloaders(a.data, batch_size=a.batch_size, calibration_size=0,
                                          image_size=model.visual.input_resolution, num_workers=a.workers)
    acc = evaluate_snn(model, pool, artifact, evaluation, resolve_class_names(evaluation.dataset), a.timesteps, a.device)
    print(f"top1@T={a.timesteps}: {acc:.6f}")

if __name__ == "__main__":
    main()
