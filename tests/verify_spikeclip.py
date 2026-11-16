"""Independent, real-data acceptance check against the training implementation.

Only this verification command imports external SpikeCLIP source. Normal artifact
loading and evaluation never need --reference-root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from CLIPs import get_clip
from clip_with_snn_visual import CLIPWithSNNVisual
from dataloader import CUB_Adapter
from eval_snn import evaluate
from prompt import build_prompt
from to_snn.coding import rate_coding
from to_snn.prototype import SNNArtifact

# Declared before observing results. FP32 reduction order differs between the
# batched reference and streaming implementation; 1e-4 permits small rounding
# effects while bounding cosine error independently of CLIP's logit scale.
ATOL = 1e-4
RTOL = 1e-4


class HierarchySubset(Subset):
    """Keep the metadata consumed by the unmodified evaluation entry point."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.class_names = dataset.class_names
        self.parent = dataset.parent


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def load_reference(root, checkpoint, device):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location('spikeclip_reference', root / 'model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(root))
    reference = module.Vision_Encoder(in_channels=3, target_dim=512,
        embed_dims=384, num_heads=8, mlp_ratios=4, depths=4, T=8,
        neuron_backend='torch')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    reference.load_state_dict(payload['state_dict'], strict=True)
    return reference.to(device).float().eval()


def errors(actual, expected):
    delta = (actual - expected).abs()
    return {'max_absolute_error': delta.max().item(),
            'mean_absolute_error': delta.mean().item()}


@torch.no_grad()
def verify(args):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ann = get_clip('openai_vit_B_16', device=args.device).float().eval()
    artifact = SNNArtifact.load(args.artifact, map_location=args.device).float().eval()
    model = CLIPWithSNNVisual(ann, artifact, rate_coding).eval()
    kwargs = {'metadata_dir': args.metadata_dir} if args.metadata_dir else {}
    dataset = CUB_Adapter(args.data, image_size=model.input_resolution, **kwargs)
    if args.samples > len(dataset) or args.samples < 1:
        raise ValueError('--samples must be between 1 and the dataset size')
    # Spread deterministically over the dataset instead of testing one class.
    indices = torch.linspace(0, len(dataset) - 1, args.samples).long().tolist()
    subset = HierarchySubset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.samples, shuffle=False, num_workers=0)
    images = next(iter(loader))['image'].to(args.device).float()
    prompts = [build_prompt(name) for name in dataset.class_names['fine_grained']]
    text = ann.encode_text(ann.tokenize(prompts).to(args.device))
    text = text.float()
    # Load and run the converted artifact before importing any external source.
    converted = model.encode_image(images, 16)
    repeated = model.encode_image(images, 8)
    model.encode_image(images.flip(0) * 0.5, 3)
    isolated = model.encode_image(images, 8)
    one_step = model.encode_image(images, 1)
    reference = load_reference(args.reference_root, args.checkpoint, args.device)
    from spikingjelly.clock_driven import functional
    functional.reset_net(reference)
    expected = reference(F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False))
    expected_cosine = F.normalize(expected, dim=-1) @ F.normalize(text, dim=-1).t()
    scale = ann.logit_scale.exp().item()
    actual_logits = model(converted[7], text)
    actual_cosine = actual_logits / scale
    checks = {
        'cosine_close': torch.allclose(actual_cosine, expected_cosine, atol=ATOL, rtol=RTOL),
        'finite': all(torch.isfinite(x).all().item() for x in
                      (converted, repeated, isolated, one_step, expected_cosine, actual_cosine)),
        'shape': list(converted.shape) == [16, args.samples, 512],
        'repeat_reset': torch.equal(repeated, isolated),
        'prefix_t8': torch.equal(converted[:8], repeated),
        'prefix_t1': torch.equal(converted[:1], one_step),
    }
    output = Path(args.output)
    smoke_dir = output.parent / (output.stem + '_eval_smoke')
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke = evaluate(model, loader, args.device, 16, args.samples,
                     smoke_dir, 1)
    checks['evaluation_smoke'] = smoke['samples'] == args.samples and smoke['timesteps'] == 16
    report = {
        'passed': all(checks.values()), 'checks': checks,
        'tolerance': {'atol': ATOL, 'rtol': RTOL, 'quantity': 'normalized cosine'},
        'cosine': errors(actual_cosine, expected_cosine),
        'scaled_logits': errors(actual_logits, expected_cosine * scale),
        'logit_scale': scale, 'reference_T': 8, 'extended_T': 16,
        'dtype': 'float32', 'tf32': False, 'device': args.device,
        'samples': args.samples, 'indices': indices,
        'sample_paths': [dataset.samples[i][0] for i in indices],
        'prompts': prompts, 'text_backend': 'untouched OpenAI ViT-B/16',
        'reference_root': str(Path(args.reference_root).resolve()),
        'reference_source_sha256': digest(Path(args.reference_root) / 'model.py'),
        'checkpoint_sha256': digest(args.checkpoint),
        'artifact_sha256': digest(args.artifact),
        'torch_version': torch.__version__, 'evaluation_smoke': smoke,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('passed', 'checks', 'cosine', 'scaled_logits')}, indent=2))
    return report['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--data', required=True, help='CUB evaluation root')
    parser.add_argument('--metadata-dir', help='Optional FreeGrainLearning/data directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args) else 1)


if __name__ == '__main__':
    main()
