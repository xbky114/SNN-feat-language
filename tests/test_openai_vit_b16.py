"""Public backend tests; pretrained downloads are replaced at the HTTP boundary."""
import urllib.request

import pytest

from CLIPs import get_clip


def test_factory_requests_original_openai_vit_b16_weights(monkeypatch, tmp_path):
    class DownloadRequested(Exception):
        pass

    def capture_download(url):
        raise DownloadRequested(url)

    monkeypatch.setattr(urllib.request, "urlopen", capture_download)
    with pytest.raises(DownloadRequested, match=r"/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt$"):
        get_clip("openai_vit_B_16", device="cpu", download_root=str(tmp_path))


def test_ann_adapter_preserves_text_features_and_public_inference_contract():
    import torch
    from CLIPs.openai_vit_B_16 import OpenAIViTB16Adapter
    from CLIPs.openai_vit_L_14.model import CLIP

    # Small real CLIP towers keep this test offline and inexpensive. The output
    # projection and tokenizer vocabulary match the B/16 public contract.
    torch.manual_seed(12)
    model = CLIP(512, 32, 1, 64, 16, 77, 49408, 64, 1, 1).eval()
    adapter = OpenAIViTB16Adapter(model, preprocessing=None)
    tokens = adapter.tokenize(["a bird", "a photo of a bird"])
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected_text = model.encode_text(tokens)
        actual_text = adapter.encode_text(tokens)
        actual_image = adapter.encode_image(images)
        image_logits, text_logits = adapter(images, tokens)
    assert tokens.shape == (2, 77)
    assert tokens[0, :4].tolist() == [49406, 320, 3329, 49407]
    assert actual_text.shape == actual_image.shape == (2, 512)
    torch.testing.assert_close(actual_text, expected_text, rtol=0, atol=0)
    assert torch.isfinite(image_logits).all()
    torch.testing.assert_close(text_logits, image_logits.t())
    assert adapter.pool is None
    assert adapter.visual is model.visual
    assert adapter.input_resolution == 32
    assert adapter.dtype == torch.float32
    assert adapter.logit_scale is model.logit_scale


def test_conversion_plan_identifies_b16_backend():
    from CLIPs.openai_vit_B_16 import OpenAIViTB16Adapter
    from CLIPs.openai_vit_L_14.model import CLIP

    model = CLIP(512, 32, 1, 64, 16, 77, 49408, 64, 1, 1).eval()
    plan = OpenAIViTB16Adapter(model, None).conversion_plan()
    assert plan.metadata["backend"] == "openai_vit_B_16"
    assert plan.metadata["text_tower"] == "ann"
