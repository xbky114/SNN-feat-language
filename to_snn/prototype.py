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
    """Self-contained visual tower shared by conversion methods."""

    def reset_state(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_state'):
                m.reset_state()

    def save(self, path):
        torch.save(self._checkpoint(), Path(path))

    def __init__(self, model, conversion_params=None, conversion_plan=None):
        super().__init__()
        self.model = model
        self.conversion_params = conversion_params
        self.conversion_plan = conversion_plan

    def forward(self, x):
        return self.model(x)

    def _checkpoint(self):
        return {'visual_model': self.model,
                'conversion_params': self.conversion_params,
                'conversion_plan': self.conversion_plan}

    @classmethod
    def load(cls, path, map_location='cpu'):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        return cls(payload['visual_model'], payload.get('conversion_params'),
                   payload.get('conversion_plan')).eval()
