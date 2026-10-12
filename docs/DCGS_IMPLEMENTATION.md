# DCGS implementation design

## Purpose

This document defines the refactoring contract for the vanilla DCGS conversion
method. DCGS is a conversion method, not an ANN model package. Its implementation
lives under `to_snn/DCGS/` and converts an ANN supplied by a model package into a
stateful SNN whose forward call processes one simulation timestep.

The model adapter covered by this design is `CLIPs/openai_rn50`. Other model
packages may implement the same ANN conversion interface later.

## Domain terms

- **ANN adapter**: the model package that constructs an ANN and describes the
  modules and parameters DCGS must convert.
- **Conversion plan**: the ANN adapter's description of conversion targets,
  thresholds, input shape, and output shape.
- **Conversion method**: an implementation such as DCGS that applies a plan and
  builds an SNN.
- **Timestep call**: one call to the converted SNN for one input tensor `x_t`.
- **SNN state**: membrane potentials and other temporal values retained between
  timestep calls.
- **Reset**: an explicit caller operation that clears all SNN state before a new
  sample or sequence.

## Runtime contract

DCGS uses single-step execution only. There is no public `step_mode` option and
no multi-step (`m`) implementation.

Evaluation owns the time loop:

```python
snn.reset_state()
outputs = []
for t in range(time_steps):
    x_t = x                         # static input repeated at every timestep
    outputs.append(snn(x_t))        # one timestep output y_t
```

The converted SNN accepts one ordinary ANN-shaped tensor per call, without a
time dimension. It returns the output for that timestep. The caller owns state
lifecycle and must call `reset_state()` before processing a new sample or
sequence.

The returned sequence is `[y_0, y_1, ..., y_(T-1)]`. DCGS does not average, sum,
or otherwise decode this sequence. Readout and metrics are separate concerns.

## Package layout

```text
CLIPs/
└── openai_rn50/
    ├── model.py
    ├── conversion.py
    └── __init__.py

to_snn/
├── __init__.py
├── prototype.py
└── DCGS/
    ├── __init__.py
    ├── config.py
    ├── converter.py
    ├── threshold.py
    └── neurons/
        ├── __init__.py
        ├── if_rate.py
        └── mth_rate.py
```

`metrics/`, `eval.py`, and other evaluation modules retain abstract interfaces
outside this refactoring document. They must consume the timestep outputs and
must not construct DCGS neurons or inspect DCGS internals.

## ANN adapter interface

`CLIPs/openai_rn50` owns model-specific knowledge. It exposes:

```python
def build_model(weights=None) -> torch.nn.Module:
    ...

def make_conversion_plan(
    model: torch.nn.Module,
    thresholds: "ThresholdTable",
) -> "ConversionPlan":
    ...
```

The plan contains named module paths and semantic target kinds. It may describe
ordinary ReLU replacements, threshold-hook replacements, and model-specific
pooling targets. It does not instantiate IF or MTH modules and does not select a
DCGS neuron.

The plan also contains `model_name`, `input_spec`, and `output_spec`. DCGS must
not branch on `openai_rn50` or any other model name.

## DCGS conversion interface

```python
@dataclass(frozen=True)
class DCGSConfig:
    neuron: Literal["IF", "MTH"]
    time_steps: int
    num_thresholds: int | None = None
```

```python
def convert(
    ann: torch.nn.Module,
    plan: ConversionPlan,
    config: DCGSConfig,
) -> SNNArtifact:
    ...
```

`convert` must:

1. validate the configuration;
2. create a non-mutating copy of the ANN;
3. calculate or load DCGS thresholds through `threshold.py`;
4. replace the plan's targets with the selected IF or MTH rate neuron;
5. attach `reset_state()` to the resulting SNN interface;
6. return a complete, forward-ready SNN artifact.

The public configuration contains only the supported DCGS choices. Differential,
negative, leaky, line, and multi-step variants are not compatibility modes and
must not remain selectable.

## Neuron behavior

`IFRate` and `MTHRate` preserve the existing DCGS timestep behavior.

```python
class DCGSNeuron(torch.nn.Module):
    def forward(self, x_t) -> torch.Tensor:
        ...

    def reset_state(self) -> None:
        ...
```

`IFRate` emits the current IF output for one timestep. `MTHRate` emits the
current multi-threshold output for one timestep, including its existing value
magnitude semantics. This refactor must not convert MTH output into binary
spikes or decode it into a rate.

Runtime state and configuration must be separate. `time_steps` is immutable
configuration; timestep counters, membrane values, and accumulators are runtime
state cleared by `reset_state()`.

## Threshold calculation

Threshold calculation belongs to `to_snn/DCGS/threshold.py`, because it is part
of conversion rather than ANN construction. The ANN adapter may expose the
locations and shape metadata needed by calibration, but DCGS owns the threshold
procedure and its result type.

```python
@dataclass(frozen=True)
class ThresholdTable:
    model_name: str
    values: dict[str, torch.Tensor]
```

Thresholds are keyed by conversion-plan target path. The converter must fail
with a clear error when a target requiring calibration has no threshold.

## Removal requirements

The supported DCGS package must contain only rate-coded IF and MTH behavior.
Remove the implementation and public references for:

- `IF_diff`, `IF_with_neg`, and related variants;
- `MTH_diff`, `MTH_with_neg`, and related variants;
- LIF variants;
- line-neuron variants;
- differential and leaky coding branches;
- `coding_type` and `step_mode` configuration;
- model-name conditionals in the generic converter;
- multi-step loops and temporal expansion/merge helpers used only by `m` mode.

Existing model-specific handling must be represented as conversion-plan target
metadata in `CLIPs/openai_rn50`, not as model-name checks in DCGS.

## Invariants and verification

The refactored modules must verify these invariants:

1. Repeated calls after `reset_state()` produce the same sequence for the same
   input and configuration.
2. Calling `reset_state()` removes membrane and timestep state from every
   converted neuron.
3. A timestep call never expects or creates a time dimension.
4. DCGS conversion does not mutate the supplied ANN.
5. IF and MTH produce the same per-timestep numerical behavior as their retained
   vanilla implementations for equivalent parameters.
6. Missing thresholds and unsupported target kinds fail during conversion, not
   during evaluation.

The implementation should test neuron state reset, plan application, threshold
lookup, non-mutating conversion, and an end-to-end `openai_rn50` conversion that
executes `T` caller-driven timestep calls.

