import copy, torch
from torch import nn
from .neurons.if_rate import IFRate
from .neurons.mth_rate import MTHRate
from .threshold import collect_thresholds
from .config import DCGSConversionParams
from to_snn.prototype import SNNArtifact
class VisualSNNArtifact(SNNArtifact):
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self,x): return self.model(x)
    def _checkpoint(self): return {'state_dict':self.state_dict()}
def convert(visual, plan, params, calibration_loader=None, thresholds=None):
    if params.neuron not in ('IF','MTH'): raise ValueError('neuron must be IF or MTH')
    if thresholds is None:
        if calibration_loader is None: raise ValueError('calibration_loader or thresholds required')
        thresholds=collect_thresholds(visual, calibration_loader, plan.targets, params.threshold_mode)
    model=copy.deepcopy(visual)
    cls=IFRate if params.neuron=='IF' else MTHRate
    for t in plan.targets:
        if t.kind not in ('relu','activation'): continue
        parent, name=t.path.rsplit('.',1) if '.' in t.path else ('',t.path); holder=model.get_submodule(parent) if parent else model
        setattr(holder,name,cls(thresholds[t.path], params.num_thresholds) if cls is MTHRate else cls(thresholds[t.path]))
    return VisualSNNArtifact(model)
