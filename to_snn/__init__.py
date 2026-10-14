from dataclasses import fields, is_dataclass
import importlib
from pathlib import Path

import yaml

from .prototype import SNNArtifact, ConversionPlan, ConversionTarget
from .coding import CODING
from .DCGS import convert
from CLIPs import get_clip


def load_conversion_params(path, method):
    """Load a method's ``ConversionParams`` dataclass from YAML.

    Conversion methods own their parameter type.  The generic loader only
    discovers ``ConversionParams`` from ``to_snn.<method>`` (or its
    ``config`` module), so adding a new method does not require changing this
    package-level API.
    """
    method_name = str(method).strip()
    module_name = method_name if method_name.startswith('to_snn.') else f'to_snn.{method_name}'
    module = importlib.import_module(module_name)
    params_type = getattr(module, 'ConversionParams', None)
    if params_type is None:
        try:
            params_type = getattr(importlib.import_module(f'{module_name}.config'),
                                  'ConversionParams')
        except (ImportError, AttributeError) as exc:
            raise TypeError(f'{module_name} must expose ConversionParams') from exc
    if not is_dataclass(params_type):
        raise TypeError(f'{module_name}.ConversionParams must be a dataclass')
    with Path(path).open('r', encoding='utf-8') as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f'conversion config must contain a mapping: {path}')
    valid = {field.name for field in fields(params_type)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f'unknown conversion parameter(s): {", ".join(unknown)}')
    return params_type(**values)

def convert_clip(clip, conversion_method, params, dataloader=None, **kwargs):
    loaded = get_clip(clip, **kwargs)
    plan = loaded.conversion_plan()
    method = __import__(f'to_snn.{conversion_method}', fromlist=['convert'])
    return method.convert(getattr(loaded, loaded.to_be_converted), plan, params, calibration_loader=dataloader)

def load_artifact(path, map_location='cpu'):
    return SNNArtifact.load(path, map_location=map_location)
