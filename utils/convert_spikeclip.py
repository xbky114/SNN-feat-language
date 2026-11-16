"""Convert the verified SpikeCLIP CUB rate-coded visual checkpoint.

Run from the repository root: python -m utils.convert_spikeclip INPUT OUTPUT.
Runtime dependency: spikingjelly==0.0.0.0.14 (standard torch backend).
Checkpoint files must be trusted, as the artifact protocol uses torch pickle.
"""
import argparse
from collections.abc import Mapping

import torch

from to_snn.prototype import SNNArtifact
from to_snn.spikeclip_visual import SpikeCLIPVisual


def convert_checkpoint(checkpoint_path, output_path):
    """Strictly load the supported visual and save a portable SNNArtifact.

    Accept a bare visual state dict or the training checkpoint's ``state_dict``.
    Other architectures, prefixes, missing tensors and extra tensors fail closed.
    """
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload) if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError('incompatible SpikeCLIP checkpoint: expected a state dictionary')
    model = SpikeCLIPVisual()
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [key for key in expected.keys() & state.keys()
                  if not isinstance(state[key], torch.Tensor)
                  or state[key].shape != expected[key].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f'incompatible SpikeCLIP checkpoint: missing={missing}, '
                         f'unexpected={unexpected}, shape_mismatch={mismatched}')
    model.load_state_dict(state, strict=True)
    artifact = SNNArtifact(model.eval(), conversion_params={
        'method': 'spikeclip', 'architecture': 'Spikingformer',
        'embed_dims': 384, 'num_heads': 8, 'depths': 4, 'mlp_ratio': 4,
        'target_dim': 512, 'training_timesteps': 8, 'coding': 'rate',
        'resize': {'size': 32, 'mode': 'bilinear', 'align_corners': False},
        'neuron': {'tau': 2.0, 'v_threshold': 1.0, 'attention_threshold': 0.5,
                   'v_reset': 0.0, 'detach_reset': True, 'decay_input': True,
                   'backend': 'torch'},
    }).eval()
    artifact.save(output_path)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output)
    print(f'Saved SpikeCLIP visual artifact: {args.output}')


if __name__ == '__main__':
    main()
