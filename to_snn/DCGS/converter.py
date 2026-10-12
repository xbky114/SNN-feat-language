import copy, torch
from torch import nn
from .neurons.if_rate import IFRate
from .neurons.mth_rate import MTHRate
from .threshold import collect_thresholds
from .config import DCGSConversionParams
from to_snn.prototype import SNNArtifact
class VisualSNNArtifact(SNNArtifact):
    def __init__(self, model, conversion_params=None, conversion_plan=None):
        super().__init__(); self.model=model
        self.conversion_params=conversion_params
        self.conversion_plan=conversion_plan
    def forward(self,x): return self.model(x)
    def _checkpoint(self):
        return {'visual_model': self.model,
                'conversion_params': self.conversion_params,
                'conversion_plan': self.conversion_plan}
    @classmethod
    def load(cls, path, map_location='cpu'):
        payload=torch.load(path, map_location=map_location, weights_only=False)
        return cls(payload['visual_model'], payload.get('conversion_params'), payload.get('conversion_plan'))
def convert(visual, plan, params, calibration_loader=None, thresholds=None):
    if params.neuron not in ('IF','MTH'): raise ValueError('neuron must be IF or MTH')
    if thresholds is None:
        if calibration_loader is None: raise ValueError('calibration_loader or thresholds required')
        thresholds=collect_thresholds(
            visual, calibration_loader, plan.targets, params.threshold_mode,
            level=params.threshold_level, time_steps=params.time_steps,
            num_thresholds=params.num_thresholds)
    model=copy.deepcopy(visual)
    cls=IFRate if params.neuron=='IF' else MTHRate
    for t in plan.targets:
        if t.kind not in ('relu','activation'): continue
        parent, name=t.path.rsplit('.',1) if '.' in t.path else ('',t.path); holder=model.get_submodule(parent) if parent else model
        setattr(holder,name,cls(thresholds[t.path], params.num_thresholds) if cls is MTHRate else cls(thresholds[t.path]))
    return VisualSNNArtifact(model, params, plan)
