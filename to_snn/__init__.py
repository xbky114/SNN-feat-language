from .prototype import SNNArtifact, ConversionPlan, ConversionTarget
from .DCGS import convert
from CLIPs import get_clip

def convert_clip(clip, conversion_method, params, dataloader=None, **kwargs):
    loaded = get_clip(clip, **kwargs)
    plan = loaded.conversion_plan()
    method = __import__(f'to_snn.{conversion_method}', fromlist=['convert'])
    return method.convert(getattr(loaded, loaded.to_be_converted), plan, params, calibration_loader=dataloader)

def load_artifact(path, map_location='cpu'):
    return SNNArtifact.load(path, map_location=map_location)
