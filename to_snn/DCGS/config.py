from dataclasses import dataclass
@dataclass(frozen=True)
class DCGSConversionParams:
    neuron: str='IF'
    time_steps: int=1
    threshold_mode: str='99.9'
    threshold_level: str='layer'
    num_thresholds: int=1

# Common method configuration name used by ``to_snn.load_conversion_params``.
ConversionParams = DCGSConversionParams
