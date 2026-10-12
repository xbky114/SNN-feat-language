from .prototype import SNNArtifact, ConversionPlan, ConversionTarget
from .DCGS import convert

def convert_clip(clip, conversion_method, params, dataloader=None, **kwargs):
    adapter = __import__(f'CLIPs.{clip}', fromlist=[''])
    loaded = adapter.load(**kwargs)
    plan = adapter.make_conversion_plan(loaded.model)
    method = __import__(f'to_snn.{conversion_method}', fromlist=['convert'])
    return method.convert(getattr(loaded.model, loaded.to_be_converted), plan, params, calibration_loader=dataloader)

def load_artifact(path, map_location='cpu'):
    return SNNArtifact.load(path, map_location=map_location)
