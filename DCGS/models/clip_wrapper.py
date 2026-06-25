import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_ann_backbone(visual, image):
    if hasattr(visual, 'forward_backbone'):
        dtype = next(visual.parameters()).dtype
        return visual.forward_backbone(image.type(dtype))

    x = image.type(visual.conv1.weight.dtype)
    x = visual.relu1(visual.bn1(visual.conv1(x)))
    x = visual.relu2(visual.bn2(visual.conv2(x)))
    x = visual.relu3(visual.bn3(visual.conv3(x)))
    x = visual.avgpool(x)
    x = visual.layer1(x)
    x = visual.layer2(x)
    x = visual.layer3(x)
    x = visual.layer4(x)
    return x


def backbone_cosine(feat_a, feat_b):
    return F.cosine_similarity(feat_a.flatten(1), feat_b.flatten(1), dim=1).item()


class CLIPWithSNNVisual(nn.Module):
    def __init__(self, snn_visual, clip_model):
        super().__init__()
        self.visual = snn_visual
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        self.context_length = clip_model.context_length

    @property
    def dtype(self):
        return next(self.visual.parameters()).dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_backbone(self, image):
        from forwards.clip_forward import encode_backbone_aggregate
        x = image.type(self.dtype)
        if hasattr(self.visual, 'init_forward_backbone'):
            return encode_backbone_aggregate(self.visual, x)
        return self.visual.forward_backbone(x)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, img_feat, text_feat):
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_feat @ text_feat.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    @torch.no_grad()
    def test_probs_at_all_t(self, image, text):
        img_features = self.encode_image(image)
        text_features = self.encode_text(text)
        T, B, _ = img_features.shape
        n_labels = text_features.shape[0]
        probs = np.zeros((B, T, n_labels), dtype=np.float32)
        for t in range(T):
            logits_per_image, _ = self.forward(img_features[t], text_features)
            probs[:, t, :] = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs
