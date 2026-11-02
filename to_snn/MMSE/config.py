"""Required parameters for layerwise minimum-MSE threshold search."""

from dataclasses import dataclass
import math

from to_snn import coding as temporal_coding


@dataclass(frozen=True)
class MMSEConversionParams:
    time_steps: int
    search_start: float
    search_num: int
    neuron: str
    num_thresholds: int
    coding: str

    def __post_init__(self):
        for name, minimum in (("time_steps", 1), ("search_num", 2), ("num_thresholds", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.search_start, bool)
                or not isinstance(self.search_start, (int, float))
                or not math.isfinite(self.search_start)
                or not 0 <= self.search_start < 1):
            raise ValueError("search_start must be finite and in [0, 1)")
        if self.neuron not in ("IF", "MTH"):
            raise ValueError("neuron must be IF or MTH")
        if (not isinstance(self.coding, str)
                or self.coding not in temporal_coding.__all__
                or not self.coding.endswith("_coding")
                or not callable(getattr(temporal_coding, self.coding, None))):
            raise ValueError(f"unknown coding function: {self.coding!r}")


ConversionParams = MMSEConversionParams
