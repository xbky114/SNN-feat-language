"""Public conversion and inference contract for the rate-coded SpikeCLIP visual."""
import pytest
import torch


def test_incompatible_checkpoint_is_rejected(tmp_path):
    from utils.convert_spikeclip import convert_checkpoint
    path = tmp_path / 'invalid.pt'
    torch.save({'state_dict': {'proj.weight': torch.ones(512, 384)}}, path)
    with pytest.raises(ValueError, match='incompatible'):
        convert_checkpoint(path, tmp_path / 'artifact.pt')
    assert not (tmp_path / 'artifact.pt').exists()


def test_saved_visual_drives_existing_wrapper_without_external_source(tmp_path):
    from CLIPs import CLIPAdapter
    from clip_with_snn_visual import CLIPWithSNNVisual
    from to_snn.coding import rate_coding
    from to_snn.prototype import SNNArtifact
    from to_snn.spikeclip_visual import SpikeCLIPVisual
    from utils.convert_spikeclip import convert_checkpoint

    class TextBoundary(CLIPAdapter):
        @property
        def pool(self):
            return None

        @property
        def logit_scale(self):
            return torch.tensor(0.0)

    # A constant projected visual has a known cosine of 1 against its bias
    # and zero against an orthogonal text. This catches lost/doubled bias and
    # wrong temporal scaling without deriving an expectation from the adapter.
    torch.set_num_threads(2)
    model = SpikeCLIPVisual()
    state = model.state_dict()
    state['proj.weight'].zero_()
    state['proj.bias'].zero_()
    state['proj.bias'][0] = 2.0
    checkpoint, output = tmp_path / 'source.pt', tmp_path / 'visual.pt'
    torch.save({'state_dict': state}, checkpoint)
    convert_checkpoint(checkpoint, output)
    wrapper = CLIPWithSNNVisual(TextBoundary(), SNNArtifact.load(output), rate_coding)
    with torch.no_grad():
        features = wrapper.encode_image(torch.randn(2, 3, 48, 48), T=10)
        scores = wrapper(features, torch.eye(512)[:2])
    assert features.shape == (10, 2, 512)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[..., 0], torch.full((10, 2), 2.0))
    torch.testing.assert_close(scores, torch.tensor([1., 0.]).expand(10, 2, 2))


def test_real_checkpoint_reference_and_batch_reset(tmp_path):
    """Opt in with SPIKECLIP_REFERENCE_ROOT and SPIKECLIP_CHECKPOINT.

    This independent training-source reference is never a runtime dependency.
    The fixed cosine tolerance is 1e-5 in CPU FP32.
    """
    import importlib.util
    import os
    import sys
    from pathlib import Path

    source = os.environ.get('SPIKECLIP_REFERENCE_ROOT')
    checkpoint = os.environ.get('SPIKECLIP_CHECKPOINT')
    if not source or not checkpoint:
        pytest.skip('set SPIKECLIP_REFERENCE_ROOT and SPIKECLIP_CHECKPOINT to opt in')
    from CLIPs import CLIPAdapter
    from clip_with_snn_visual import CLIPWithSNNVisual
    from to_snn.coding import rate_coding
    from to_snn.prototype import SNNArtifact
    from utils.convert_spikeclip import convert_checkpoint
    from spikingjelly.clock_driven import functional

    class TextBoundary(CLIPAdapter):
        @property
        def pool(self):
            return None

        @property
        def logit_scale(self):
            return torch.tensor(0.0)

    torch.set_num_threads(2)
    sys.path.insert(0, source)
    try:
        spec = importlib.util.spec_from_file_location('spikeclip_reference', Path(source) / 'model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(source)
    reference = module.Vision_Encoder(num_heads=8, T=8, neuron_backend='torch').eval()
    reference.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'], strict=True)
    output = tmp_path / 'visual.pt'
    convert_checkpoint(checkpoint, output)
    wrapper = CLIPWithSNNVisual(TextBoundary(), SNNArtifact.load(output), rate_coding)
    generator = torch.Generator().manual_seed(123)
    images = torch.randn(2, 3, 48, 48, generator=generator)
    texts = torch.randn(7, 512, generator=generator)
    with torch.no_grad():
        functional.reset_net(reference)
        expected = reference(torch.nn.functional.interpolate(images, (32, 32), mode='bilinear', align_corners=False))
        features = wrapper.encode_image(images, 10)
        torch.testing.assert_close(wrapper(features[7], texts), wrapper(expected, texts), atol=1e-5, rtol=0)
        wrapper.encode_image(torch.randn(1, 3, 48, 48), 3)
        repeated = wrapper.encode_image(images, 10)
        torch.testing.assert_close(features, repeated, atol=0, rtol=0)
        # The ninth/tenth step must continue state rather than cycle at T=8.
        reference.T = 10
        functional.reset_net(reference)
        expected10 = reference(torch.nn.functional.interpolate(images, (32, 32), mode='bilinear', align_corners=False))
        torch.testing.assert_close(wrapper(features[9], texts), wrapper(expected10, texts), atol=1e-5, rtol=0)
