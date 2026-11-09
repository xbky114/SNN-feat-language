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
        if not isinstance(coding, CODING):
            raise TypeError("coding must be a CODING instance")
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

    @property
    def pool_dtype(self):
        return self._clip.pool_dtype

    def encode_image(self, img, T):
        if T <= 0:
            raise ValueError("T must be positive")
        self.visual.reset_state()
        coded = None
        image_features = []
        for t in range(1, T + 1):
            coded = self.coding.update(coded, self.visual(img), t)
            # Keep the encoder's state precision; cast only the pool input.
            if self.pool is None:
                # ViT visual towers already return their projected class-token
                # feature and have no separate pooling module.
                image_features.append(coded)
            else:
                image_features.append(self.pool(coded.to(dtype=self.pool_dtype)))
        return torch.stack(image_features, dim=0)

    def forward(self, img_feat, text_feat):
        image_features = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_features = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # ECMT ViT artifacts run their recurrent visual tower in FP32 for
        # numerical stability, while the OpenAI text tower may remain FP16.
        # Promote the small text matrix at the final boundary so mixed-dtype
        # matmul does not fail and image/text normalization uses one dtype.
        text_features = text_features.to(dtype=image_features.dtype)
        return self.logit_scale.exp() * image_features @ text_features.t()

    def calc_logits_through_time(self, img, texts, T):
        image_features = self.encode_image(img, T)
        return self.calc_logits_from_image_features(image_features, texts)

    def calc_logits_from_image_features(self, image_features, texts):
        """Score texts against reusable image features shaped [T, batch, dim]."""
        text_features = self.encode_text(self.tokenize(texts).to(image_features.device))
        return torch.stack(
            [self(features, text_features) for features in image_features],
            dim=0,
        )


__all__ = ["CLIPWithSNNVisual"]
