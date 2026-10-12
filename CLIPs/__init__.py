"""CLIP adapter definitions and the model factory."""
from importlib import import_module
import re
from torch import nn


class CLIPAdapter(nn.Module):
    """Common interface implemented by the supported CLIP backends."""
    to_be_converted='visual'

    def __init__(self):
        super().__init__()

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
    def conversion_plan(self): raise NotImplementedError


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
