from CLIPs import CLIPAdapter
from torch import nn
from to_snn.prototype import ConversionPlan, ConversionTarget

class CNNBackbone(nn.Module):
    """RN101 convolutional tower ending before attention pooling."""
    def __init__(self, visual):
        super().__init__()
        self.source = visual

    def forward(self, x):
        return self.source.forward_backbone(x)

class OpenAIRN101Adapter(CLIPAdapter):
    to_be_converted='visual'
    def __init__(self, model, visual, pool, preprocessing=None):
        super().__init__(); self._model=model; self._visual=visual; self._pool=pool; self._preprocessing=preprocessing
    @property
    def model(self): return self._model
    @property
    def preprocessing(self): return self._preprocessing
    @property
    def visual(self): return self._visual
    @property
    def pool(self): return self._pool
    def conversion_plan(self):
        return make_conversion_plan(self.visual)
    def encode_image(self, image):
        return self._model.encode_image(image)

    def encode_text(self, text):
        return self._model.encode_text(text)

    def tokenize(self, texts, context_length=77, truncate=False):
        from .clip import tokenize
        return tokenize(texts, context_length=context_length, truncate=truncate)

    @property
    def logit_scale(self):
        return self._model.logit_scale

    @property
    def input_resolution(self):
        return self._model.visual.input_resolution

    @property
    def dtype(self):
        return self._model.dtype

    @property
    def pool_dtype(self):
        # OpenAI CLIP can keep the CNN backbone in fp32 while its attention
        # pool projections use the model's reduced-precision dtype.
        q_proj = getattr(self._pool, 'q_proj', None)
        if q_proj is not None and hasattr(q_proj, 'weight'):
            return q_proj.weight.dtype
        try:
            return next(self._pool.parameters()).dtype
        except StopIteration:
            return self.dtype

    
def make_conversion_plan(model):
    root=model.visual if hasattr(model, 'visual') else model
    targets=[]
    for n,m in root.named_modules():
        if n and isinstance(m, __import__('torch').nn.ReLU): targets.append(ConversionTarget(n,'relu'))
    return ConversionPlan(tuple(targets), {'tower':'visual','text_tower':'ann'})
def load(device='cpu'):
    """Load the complete OpenAI RN101 CLIP model using the bundled loader."""
    from .clip import load as openai_load
    model, preprocessing = openai_load('RN101', device=device, jit=False)
    model = model.eval()
    visual = CNNBackbone(model.visual).to(device).eval()
    return OpenAIRN101Adapter(model, visual, model.visual.attnpool, preprocessing)
