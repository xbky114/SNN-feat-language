"""Configuration for ECMT transformer conversion."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ECMTConversionParams:
    neuron: str = "TwoSide"
    time_steps: int = 8
    percent: float = 0.999
    linear_num: int = 8
    qkv_num: int = 8
    softmax_num: int = 8
    softmax_p: float = 0.0125
    gap: float = 2.0
    fused: bool = True

    def __post_init__(self):
        if self.neuron != "TwoSide":
            raise ValueError("neuron must be 'TwoSide'")
        if type(self.time_steps) is not int or self.time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        if isinstance(self.percent, bool) or not isinstance(self.percent, (int, float)) or not math.isfinite(self.percent) or not .5 < self.percent < 1:
            raise ValueError("percent must be finite and strictly between 0.5 and 1")
        for name in ("linear_num", "qkv_num", "softmax_num"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("softmax_p", "gap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if type(self.fused) is not bool:
            raise ValueError("fused must be a boolean")

ConversionParams = ECMTConversionParams
