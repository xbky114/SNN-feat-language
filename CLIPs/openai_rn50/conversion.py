from CLIPs import CLIPAdapter
from to_snn.prototype import ConversionPlan, ConversionTarget
class OpenAIRN50Adapter(CLIPAdapter):
    to_be_converted='visual'
    def __init__(self, model, preprocessing=None): self._model=model; self._preprocessing=preprocessing
    @property
    def model(self): return self._model
    @property
    def preprocessing(self): return self._preprocessing
    def conversion_plan(self):
        return make_conversion_plan(self.model)
def make_conversion_plan(model):
    root=model.visual
    targets=[]
    for n,m in root.named_modules():
        if n and isinstance(m, __import__('torch').nn.ReLU): targets.append(ConversionTarget(n,'relu'))
    return ConversionPlan(tuple(targets), {'tower':'visual','text_tower':'ann'})
def load(weights=None, device='cpu'):
    from CLIP.clip.model import build_model
    if weights is None: raise ValueError('weights must be a CLIP state dict')
    model=build_model(weights).to(device).eval()
    return OpenAIRN50Adapter(model)
