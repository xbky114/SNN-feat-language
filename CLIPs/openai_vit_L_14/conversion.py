"""OpenAI CLIP ViT-L/14 adapter and its explicit ECMT conversion plan."""
from torch import nn
from CLIPs import CLIPAdapter
from to_snn.prototype import ConversionPlan, ConversionTarget


class OpenAIViTL14Adapter(CLIPAdapter):
    to_be_converted = "visual"
    def __init__(self, model, preprocessing):
        super().__init__(); self._model = model; self._preprocessing = preprocessing
    @property
    def model(self): return self._model
    @property
    def preprocessing(self): return self._preprocessing
    @property
    def visual(self): return self._model.visual
    @property
    def pool(self): return None
    def conversion_plan(self): return make_conversion_plan(self.visual)
    def encode_image(self, image): return self._model.encode_image(image)
    def encode_text(self, text): return self._model.encode_text(text)
    def tokenize(self, texts, context_length=77, truncate=False):
        from .clip import tokenize
        return tokenize(texts, context_length=context_length, truncate=truncate)
    @property
    def logit_scale(self): return self._model.logit_scale
    @property
    def input_resolution(self): return self.visual.input_resolution
    @property
    def dtype(self): return self._model.dtype


def _add(targets, root, path, kind):
    try:
        module = root.get_submodule(path)
    except AttributeError as exc:
        raise ValueError(f"OpenAI ViT-L/14 model is missing ECMT seam {path!r}") from exc
    if not isinstance(module, nn.Module):
        raise ValueError(f"ECMT seam {path!r} is not a module")
    targets.append(ConversionTarget(path, kind))


def make_conversion_plan(model):
    """Enumerate every ECMT seam in the visual transformer.

    The plan is explicit and ordered by transformer block.  ECMT never scans
    unknown modules or silently skips an operation.
    """
    root = model.visual if hasattr(model, "visual") else model
    if not hasattr(root, "transformer") or not hasattr(root.transformer, "resblocks"):
        raise ValueError("OpenAI ViT-L/14 visual tower must expose transformer.resblocks")
    targets = []
    _add(targets, root, "ln_pre", "layernorm")
    for index, _block in enumerate(root.transformer.resblocks):
        prefix = f"transformer.resblocks.{index}"
        _add(targets, root, f"{prefix}.ln_1", "layernorm")
        _add(targets, root, f"{prefix}.attn.qkv_neuron", "qkv")
        _add(targets, root, f"{prefix}.attn.q_neuron", "q")
        _add(targets, root, f"{prefix}.attn.k_neuron", "k")
        _add(targets, root, f"{prefix}.attn.v_neuron", "v")
        _add(targets, root, f"{prefix}.attn.qk", "qk")
        _add(targets, root, f"{prefix}.attn.softmax", "softmax")
        _add(targets, root, f"{prefix}.attn.softmax_neuron", "softmax_neuron")
        _add(targets, root, f"{prefix}.attn.av", "av")
        _add(targets, root, f"{prefix}.attn.x_out_neuron", "linear")
        _add(targets, root, f"{prefix}.mlp.c_fc_neuron", "linear")
        _add(targets, root, f"{prefix}.mlp.gelu", "quickgelu")
        _add(targets, root, f"{prefix}.mlp.c_proj_neuron", "linear")
        _add(targets, root, f"{prefix}.ln_2", "layernorm")
    required = ("layernorm", "qkv", "q", "k", "v", "qk", "softmax",
                "softmax_neuron", "av", "linear", "quickgelu")
    metadata = {"tower": "visual", "text_tower": "ann", "backend": "openai_vit_L_14",
                "required_kinds": required, "attention": "explicit_qk_av"}
    return ConversionPlan(tuple(targets), metadata)


def load(device="cpu", download_root=None):
    from .clip import load as openai_load
    model, preprocessing = openai_load("ViT-L/14", device=device, jit=False, download_root=download_root)
    model.eval()
    return OpenAIViTL14Adapter(model, preprocessing)
