"""OpenAI CLIP ViT-L/14 adapter and conversion plan."""

from torch import nn

from CLIPs import CLIPAdapter
from to_snn.prototype import ConversionPlan, ConversionTarget


class OpenAIViTL14Adapter(CLIPAdapter):
    """Adapter exposing OpenAI CLIP's ViT-L/14 visual tower."""

    to_be_converted = "visual"

    def __init__(self, model, preprocessing):
        super().__init__()
        self._model = model
        self._preprocessing = preprocessing

    @property
    def model(self):
        return self._model

    @property
    def preprocessing(self):
        return self._preprocessing

    @property
    def visual(self):
        return self._model.visual

    @property
    def pool(self):
        # ViT-L/14 uses its class token instead of a separate pooling module.
        return None

    def conversion_plan(self):
        return make_conversion_plan(self.visual)

    def encode_image(self, image):
        return self._model.encode_image(image)

    def encode_text(self, text):
        return self._model.encode_text(text)

    def tokenize(self, texts, context_length=77, truncate=False):
        from .clip import tokenize
        return tokenize(texts, context_length=context_length, truncate=truncate)

    @property
    def logit_scale(self):
        return self._model.logit_scale

    @property
    def input_resolution(self):
        return self.visual.input_resolution

    @property
    def dtype(self):
        return self._model.dtype


def make_conversion_plan(model):
    """Plan replacement of FFN GELUs in the supplied visual transformer only."""
    root = model.visual if hasattr(model, "visual") else model
    targets = tuple(
        ConversionTarget(name, "gelu")
        for name, module in root.named_modules()
        if name and isinstance(module, nn.Module)
        and module.__class__.__name__ == "QuickGELU"
    )
    return ConversionPlan(targets, {"tower": "visual", "text_tower": "ann", "activations": "ffn_gelu"})


def load(device="cpu", download_root=None):
    """Load the complete OpenAI CLIP ViT-L/14 model."""
    from .clip import load as openai_load
    model, preprocessing = openai_load(
        "ViT-L/14", device=device, jit=False, download_root=download_root
    )
    model.eval()
    return OpenAIViTL14Adapter(model, preprocessing)
