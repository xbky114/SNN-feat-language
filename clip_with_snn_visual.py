"""CLIP inference wrapper using an SNN visual tower."""

import torch

from CLIPs import CLIPAdapter
from to_snn import CODING
from to_snn.prototype import SNNArtifact


class CLIPWithSNNVisual(CLIPAdapter):
    """Keep a CLIP text tower while replacing its visual tower with an SNN."""

    def __init__(self, clip: CLIPAdapter, visual: SNNArtifact, coding: CODING):
        super().__init__()
        if not isinstance(clip, CLIPAdapter):
            raise TypeError("clip must be a CLIPAdapter")
        if not isinstance(visual, SNNArtifact):
            raise TypeError("visual must be an SNNArtifact")
        self._clip = clip
        self._visual = visual
        self.coding = coding

    @property
    def model(self):
        return self

    @property
    def preprocessing(self):
        return self._clip.preprocessing

    @property
    def visual(self):
        return self._visual

    @property
    def pool(self):
        return self._clip.pool

    def encode_text(self, text):
        return self._clip.encode_text(text)

    def tokenize(self, texts, context_length=77, truncate=False):
        return self._clip.tokenize(texts, context_length, truncate)

    @property
    def logit_scale(self):
        return self._clip.logit_scale

    @property
    def input_resolution(self):
        return self._clip.input_resolution

    @property
    def dtype(self):
        return self._clip.dtype

    def encode_image(self, img, T):
        if T <= 0:
            raise ValueError("T must be positive")
        self.visual.reset_state()
        backbone_features = []
        for _ in range(T):
            backbone_features.append(self.visual(img))
        return torch.stack(
            [
                self.pool(self.coding(backbone_features[:t]))
                for t in range(1, T + 1)
            ],
            dim=0,
        )

    def forward(self, img_feat, text_feat):
        image_features = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_features = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_features @ text_features.t()

    def calc_logits_through_time(self, img, texts, T):
        image_features = self.encode_image(img, T)
        text_features = self.encode_text(self.tokenize(texts))
        return torch.stack(
            [self(image_features[t], text_features) for t in range(T)],
            dim=0,
        )


__all__ = ["CLIPWithSNNVisual"]
