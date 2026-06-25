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
        out = self.visual(image.type(self.dtype))
        if out.dim() == 3:
            out = out.mean(dim=0)
        return out

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

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
