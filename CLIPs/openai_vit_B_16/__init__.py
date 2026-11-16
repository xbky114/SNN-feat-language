"""Original OpenAI CLIP ViT-B/16, using the shared vendored CLIP loader."""

from CLIPs.openai_vit_L_14.conversion import OpenAIViTL14Adapter


class OpenAIViTB16Adapter(OpenAIViTL14Adapter):
    """ViT-B/16 adapter with the original ANN text tower and 512-D outputs."""

    backend_name = "openai_vit_B_16"


def load(device="cpu", download_root=None):
    """Load the original OpenAI ViT-B/16 weights, including its ANN text tower."""
    from CLIPs.openai_vit_L_14.clip import load as openai_load

    model, preprocessing = openai_load(
        "ViT-B/16", device=device, jit=False, download_root=download_root
    )
    model.eval()
    return OpenAIViTB16Adapter(model, preprocessing)


__all__ = ["OpenAIViTB16Adapter", "load"]
