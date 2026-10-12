
- ANN package describes conversion-relevant structure.
- Converter consumes that description and produces a complete SNN artifact.
- Evaluation consumes only the artifact interface.
- Metrics consume evaluation records, not model internals.

Use a package name such as clips/ rather than CLIPs/; lowercase package names are easier to import consistently.

project/
├── clips/
│   ├── clip_rn50/
│   │   ├── model.py
│   │   ├── conversion.py
│   │   └── __init__.py
│   └── clip_vit/
│       └── ...
├── to_snn/
│   ├── if_rate/
│   │   ├── converter.py
│   │   └── __init__.py
│   ├── lif/
│   │   └── ...
│   ├── artifact.py
│   └── __init__.py
├── metrics/
│   ├── accuracy.py
│   ├── temporal.py
│   └── __init__.py
├── eval.py
└── schemas.py

### 1. ANN modules should expose a conversion description

Do not make converters search for "relu" or special-case model names. The ANN should provide a small, declarative interface:

# clips/clip_rn50/conversion.py
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class ConversionTarget:
    path: str
    kind: str                 # "relu", "threshold_hook", "attention_pool"
    threshold: object | None = None
    metadata: dict | None = None

@dataclass(frozen=True)
class ConversionPlan:
    model_name: str
    targets: tuple[ConversionTarget, ...]
    input_shape: tuple[int, ...]
    output_spec: dict

def conversion_plan(model) -> ConversionPlan:
    ...

The ANN package can construct this plan using named module paths, explicit metadata, or torch.fx. The important point is that the conversion package receives semantic targets rather than inspecting arbitrary implementation details.

The ANN package can also provide a stable factory:

# clips/clip_rn50/__init__.py
from .model import build_model
from .conversion import conversion_plan

MODEL_NAME = "clip_rn50"

__all__ = ["MODEL_NAME", "build_model", "conversion_plan"]

This creates a clean seam:

ann = clips.clip_rn50.build_model(weights)
plan = clips.clip_rn50.conversion_plan(ann)
snn = to_snn.if_rate.convert(ann, plan, config)

The plan is part of the ANN adapter. The generic converter knows how to replace ConversionTarget(kind="relu"); it does not know what a CLIP ResNet looks like.

### 2. Keep converter interfaces small

Each conversion method should expose one primary operation:

# to_snn/protocol.py
from typing import Protocol

class Converter(Protocol):
    name: str

    def convert(self, ann, plan, config) -> "SNNArtifact":
        ...

For example:

# to_snn/if_rate/__init__.py
from .converter import convert

METHOD_NAME = "if_rate"

__all__ = ["METHOD_NAME", "convert"]

def convert(ann, plan, config) -> SNNArtifact:
    model = copy.deepcopy(ann)
    # Apply plan.targets here.
    # No model-name-specific ReLU search.
    return SNNArtifact.from_model(
        model=model,
        method=METHOD_NAME,
        config=config,
        input_spec=plan.input_shape,
        output_spec=plan.output_spec,
    )

Use copy.deepcopy or a clearly documented non-mutating conversion contract. In-place mutation of the ANN makes repeated experiments and tests difficult.

### 3. Make the SNN artifact the handoff object

The converter should return a complete, forward-ready artifact, not merely a modified model plus initialization instructions.

# to_snn/artifact.py
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class SNNArtifact:
    model: torch.nn.Module
    model_name: str
    method: str
    config: dict
    input_spec: dict
    output_spec: dict
    format_version: int = 1

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save(self, path: str | Path):
        ...

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "SNNArtifact":
        ...

Prefer saving:

artifact/
├── manifest.json
├── weights.pt
└── code_version.txt

The manifest should include:

{
  "format_version": 1,
  "model_name": "clip_rn50",
  "conversion_method": "if_rate",
  "method_config": {
    "time_steps": 8,
    "step_mode": "s"
  },
  "input_spec": {"shape": [3, 224, 224]},
  "output_spec": {"kind": "image_features"}
}

There are two reasonable serialization strategies:

1. Self-contained scripted artifact (torch.jit.save or torch.export)
   Easiest for evaluation and deployment, but less flexible.
2. Manifest + state dict + registered model factory
   More maintainable and safer across code changes. load() uses (model_name, method) to reconstruct the module, then loads weights.

For your goal, I would support both: use a self-contained artifact for experiments and a manifest/state-dict format for long-term reproducibility.

### 4. Define the SNN prototype around evaluation behavior

Evaluation should not know how an SNN was initialized. Define the minimum behavior required by evaluation:

# schemas.py
from typing import Protocol

class SNN(Protocol):
    def __call__(self, x): ...

    @property
    def time_steps(self) -> int: ...

    @property
    def output_spec(self) -> dict: ...

    def reset_state(self) -> None: ...

reset_state() matters if neurons retain membrane state. Evaluation should call it between samples or batches according to one documented lifecycle rule.

You may also want a standard output object:

@dataclass
class SNNOutput:
    values: torch.Tensor
    time_steps: int
    metadata: dict

Then evaluation can handle rate-coded, temporal, and static outputs uniformly.

### 5. Keep eval.py artifact-oriented

eval.py should depend only on the artifact and a dataset adapter:

def evaluate(artifact, dataloader, metrics):
    artifact.model.eval()

    records = []
    for batch in dataloader:
        artifact.model.reset_state()
        output = artifact.model(batch["input"])
        records.append({
            "prediction": output,
            "target": batch.get("target"),
            "metadata": batch.get("metadata"),
        })

    return compute_metrics(records, metrics)

A command-line entry point can load the artifact without importing a CLIP implementation:

python eval.py \
  --artifact runs/clip_rn50_if_rate \
  --dataset imagenet \
  --metrics accuracy temporal_energy

This gives evaluation a deep interface: “load an artifact and run it,” rather than “know how to construct every SNN.”

### 6. Metrics should consume normalized records

Metrics should not inspect neuron classes or model internals.

# metrics/protocol.py
from typing import Protocol

class Metric(Protocol):
    name: str

    def update(self, record: dict) -> None:
        ...

    def compute(self) -> dict:
        ...

Examples:

accuracy.update({
    "prediction": logits,
    "target": labels,
})

temporal_energy.update({
    "spikes": spikes,
    "time_steps": 8,
})

If metrics need temporal data, make that part of the standard output schema. Do not have each metric reach into a particular SNN implementation.

### 7. Put registries at package seams

Avoid large if model_name == ... and if method == ... blocks. Use registries:

MODEL_REGISTRY = {
    "clip_rn50": clips.clip_rn50,
    "clip_vit": clips.clip_vit,
}

METHOD_REGISTRY = {
    "if_rate": to_snn.if_rate,
    "lif": to_snn.lif,
}

SNNArtifact.load() can resolve these registries from the manifest. Adding a model or conversion method then requires implementing one adapter and registering it, rather than editing evaluation code.

The resulting dependency direction should be:

clips/*  ───────►  schemas
to_snn/* ───────►  schemas
eval.py  ───────►  schemas + artifact
metrics/ ───────►  normalized evaluation records

eval.py should not import clips.* or to_snn.* directly except through artifact loading. This preserves locality: conversion logic stays in conversion modules, model-specific knowledge stays in ANN adapters, and evaluation remains stable as new models and methods are added.