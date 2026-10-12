"""Zero-shot ImageNet evaluation for the OpenAI CLIP model."""
from __future__ import annotations
import argparse
import torch
from dataloader import imagenet_dataloaders, resolve_class_names

def class_text_features(model, class_names, device):
    from CLIPs.openai_rn50.clip import tokenize
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
    with torch.no_grad():
        text = model.encode_text(tokenize(prompts).to(device))
    return text / text.norm(dim=-1, keepdim=True)

@torch.no_grad()
def evaluate(model, loader, class_names, device="cpu"):
    model.eval().to(device)
    text_features = class_text_features(model, class_names, device)
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        pred = (model.logit_scale.exp() * features @ text_features.T).argmax(1)
        correct += (pred == labels).sum().item(); total += labels.numel()
    return correct / total if total else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--clip", required=True)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    from CLIPs import get_clip
    adapter = get_clip(a.clip, device=a.device)
    model = adapter.model
    evaluation, _ = imagenet_dataloaders(a.data, batch_size=a.batch_size, calibration_size=0,
                                          num_workers=a.workers, image_size=model.visual.input_resolution)
    print(f"top1={evaluate(model, evaluation, resolve_class_names(evaluation.dataset), a.device):.6f}")

if __name__ == "__main__":
    main()
