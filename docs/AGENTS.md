# Core implementation

## Purpose

This project studies how CLIP visual representations change when the visual
tower is converted from an artificial neural network (ANN) into a rate-coded
spiking neural network (SNN). The text tower remains an ANN. A converted
artifact therefore represents the visual tower only.

The downstream goal is semantic properties of SNN.

For a static image, evaluation supplies the same preprocessed tensor at every
timestep:

```python
snn.reset_state()
features = [snn(image) for _ in range(T)]
```

Each call returns the visual tower's final feature for one timestep.

## Design philosophy

`CLIPs/{name}` describes an ANN implementation. It loads weights, exposes the
visual tower, provides preprocessing, and describes which ANN modules may be
replaced. It does not choose an SNN neuron, timestep count, or threshold
calibration rule.

`to_snn/{method}` implements conversion. It receives an ANN visual tower (except nonlinear pooling at the end), an
ANN-only conversion plan, conversion parameters, and calibration data or saved
statistics. It computes thresholds, replaces planned modules, and returns a
self-contained visual `SNNArtifact`.

The boundary is the selected visual module, its `ConversionPlan`, and a
tensor-in/tensor-out interface. Preprocessing stays outside both objects:
adapters expose a transform, while dataloaders apply it and yield normalized
image tensors. Calibration and evaluation therefore use identical inputs.

## Public CLIP APIs

```python
from CLIPs import get_clip
clip = get_clip("openai_rn50", device="cuda")
```

`get_clip()` imports the named backend, calls `load()`, and verifies its
`CLIPAdapter` contract:

```python
clip.model             # complete loaded ANN CLIP
clip.visual            # visual module passed to conversion
clip.pool              # visual pooling component
clip.preprocessing     # external preprocessing callable
clip.to_be_converted   # currently "visual"
clip.conversion_plan() # ANN replacement targets
```

The OpenAI RN50 adapter loads the bundled CLIP implementation and exposes a
CNN visual backbone. Its plan discovers ReLU modules in that visual tower.

## Public conversion APIs

```python
from to_snn import convert_clip
from to_snn.DCGS import DCGSConversionParams

artifact = convert_clip(
    "openai_rn50", "DCGS",
    DCGSConversionParams(neuron="IF", time_steps=32),
    dataloader=calibration_loader,
    device="cuda",
)
```

`convert_clip()` loads the adapter, obtains its plan, extracts only
`to_be_converted`, and dispatches to the selected method. The text tower is
never passed to conversion.

`to_snn.DCGS` exposes `DCGSConversionParams`, `convert`, the compatibility
alias `Convert_by_DCGS`, `VisualSNNArtifact`, threshold collection and
statistics save/load helpers. Calibration consumes image-only dataloaders and
stores automatically named caches under `to_snn/DCGS/`, using the model,
statistical method, percentile when applicable, and UTC timestamp.

The current DCGS scope is single-step vanilla rate coding with IF and MTH
neurons. Differential, negative, leaky, and multi-step modes are excluded.

## Artifact lifecycle

```python
artifact.reset_state()
y_t = artifact(image_step)
artifact.save(path)
loaded = artifact.load(path)
```

The checkpoint stores the converted visual model, `DCGSConversionParams`, and
the `ConversionPlan`. The converted model contains neuron modules and their
registered threshold buffers, so loading does not require the original ANN,
calibration dataloader, or threshold cache. The ANN text tower is retained by
the caller when semantic CLIP image/text evaluation is needed.
