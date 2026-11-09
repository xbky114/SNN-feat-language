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
        # Visual checkpoints may be stored in reduced precision (OpenAI CLIP
        # commonly uses FP16 on CUDA), while dataloaders generally yield
        # FP32 tensors.  Keep the artifact boundary tensor-in/tensor-out and
        # match the converted model's placement and input dtype here.
        state = next(iter(self.model.parameters()), None)
        state = state if state is not None else next(iter(self.model.buffers()), None)
        if state is not None:
            if state.is_floating_point() and x.is_floating_point():
                x = x.to(device=state.device, dtype=state.dtype)
            else:
                x = x.to(device=state.device)
        return self.model(x)

    def _checkpoint(self):
        return {'visual_model': self.model,
                'conversion_params': self.conversion_params,
                'conversion_plan': self.conversion_plan}

    @classmethod
    def load(cls, path, map_location='cpu'):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        artifact = cls(payload['visual_model'], payload.get('conversion_params'),
                        payload.get('conversion_plan')).eval()
        # Runtime state belongs to the sequence that produced the checkpoint,
        # never to the next evaluation sequence.
        artifact.reset_state()
        return artifact
