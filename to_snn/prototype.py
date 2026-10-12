from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import torch
from torch import nn

@dataclass(frozen=True)
class ConversionTarget:
    path: str
    kind: str = 'relu'

@dataclass(frozen=True)
class ConversionPlan:
    targets: tuple[ConversionTarget, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

class SNNArtifact(nn.Module):
    def reset_state(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_state'):
                m.reset_state()

    def save(self, path):
        torch.save(self, Path(path))

    @classmethod
    def load(cls, path, map_location='cpu'):
        obj = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(obj, cls):
            raise TypeError(f'expected {cls.__name__}, got {type(obj).__name__}')
        return obj

    def _checkpoint(self):
        return {'class': type(self).__module__ + '.' + type(self).__qualname__, 'state_dict': self.state_dict()}
