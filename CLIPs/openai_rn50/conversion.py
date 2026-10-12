from CLIPs import CLIPAdapter
from torch import nn
from to_snn.prototype import ConversionPlan, ConversionTarget

class CNNBackbone(nn.Module):
    """RN50 convolutional tower ending before attention pooling."""
    def __init__(self, visual):
        super().__init__()
        self.source = visual

    def forward(self, x):
        return self.source.forward_backbone(x)

class OpenAIRN50Adapter(CLIPAdapter):
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
def make_conversion_plan(model):
    root=model.visual if hasattr(model, 'visual') else model
    targets=[]
    for n,m in root.named_modules():
        if n and isinstance(m, __import__('torch').nn.ReLU): targets.append(ConversionTarget(n,'relu'))
    return ConversionPlan(tuple(targets), {'tower':'visual','text_tower':'ann'})
def load(device='cpu'):
    """Load the complete OpenAI RN50 CLIP model using the bundled loader."""
    from .clip import load as openai_load
    model, preprocessing = openai_load('RN50', device=device, jit=False)
    model = model.eval()
    visual = CNNBackbone(model.visual).to(device).eval()
    return OpenAIRN50Adapter(model, visual, model.visual.attnpool, preprocessing)
