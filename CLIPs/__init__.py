"""CLIP adapter definitions and the model factory."""
from importlib import import_module
import re
from torch import nn


class CLIPAdapter(nn.Module):
    """Common interface implemented by the supported CLIP backends."""
    to_be_converted='visual'

    def __init__(self):
        super().__init__()

    # neural network modules
    @property
    def model(self):
        raise NotImplementedError
    @property
    def preprocessing(self):
        return None
    @property
    def visual(self):
        raise NotImplementedError
    @property
    def pool(self):
        raise NotImplementedError

    # expose the module names to be replaced by Spiking neurons.
    def conversion_plan(self): raise NotImplementedError

    # Inference interface shared by CLIP backends.  Keeping these methods on
    # the adapter means callers never need to depend on a backend's model
    # implementation or tokenizer module.
    def encode_image(self, image):
        raise NotImplementedError

    def encode_text(self, text):
        raise NotImplementedError

    def tokenize(self, texts, context_length=77, truncate=False):
        raise NotImplementedError

    @property
    def logit_scale(self):
        raise NotImplementedError

    @property
    def input_resolution(self):
        raise NotImplementedError

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def pool_dtype(self):
        """dtype expected by the visual attention-pooling layer."""
        return self.dtype

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()


def get_clip(name: str, **kwargs) -> nn.Module:
    """Load and return a named :class:`CLIPAdapter`.

    ``name`` is the backend directory name, for example ``openai_rn50``.
    Remaining keyword arguments are forwarded to that backend's ``load``
    function (typically ``weights=...`` and ``device=...``).
    """
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]+', name):
        raise ValueError(f"invalid CLIP name: {name!r}")
    try:
        backend = import_module(f'CLIPs.{name}')
    except ModuleNotFoundError as exc:
        raise ValueError(f"unknown CLIP backend: {name!r}") from exc
    loader = getattr(backend, 'load', None)
    if loader is None:
        raise ValueError(f"CLIP backend {name!r} does not expose load()")
    adapter = loader(**kwargs)
    if not isinstance(adapter, CLIPAdapter):
        raise TypeError(f"CLIP backend {name!r} returned {type(adapter).__name__}, expected CLIPAdapter")
    return adapter


__all__ = ['CLIPAdapter', 'get_clip']
