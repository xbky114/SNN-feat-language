from dataclasses import dataclass
from typing import Any
class CLIPAdapter:
    to_be_converted='visual'
    @property
    def model(self):
        raise NotImplementedError
    @property
    def preprocessing(self):
        return None
    def conversion_plan(self): raise NotImplementedError
