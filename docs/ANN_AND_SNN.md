# Core implementation

## Purpose

This project studies how CLIP visual representations change when the visual
tower (currently only CNN backbone; the conversion of nonlinear pool is a future issue) 
is converted from an artificial neural network (ANN) into a rate-coded
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
clip.conversion_plan() # ANN replacement targets, usually all 'ReLU'
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

DCGS and MMSE reject registered MaxPool (including adaptive/fractional),
LayerNorm, GELU, Softmax and attention modules before calibration, including
the custom QuickGELU/AttentionPool classes used by CLIP.

```python
from to_snn.DCGS import collect_statistics, save_statistics, load_statistics, convert

statistics = collect_statistics(
    clip.visual, calibration_loader, clip.conversion_plan().targets,
    mode="var", level="channel",
)
save_statistics(statistics, "calibration/rn50_var_channel")
statistics = load_statistics("calibration/rn50_var_channel")
artifact = convert(clip.visual, clip.conversion_plan(), params, statistics=statistics)
```

Methods may return either an `SNNArtifact` or `(artifact, info)`. The first
output must be an `SNNArtifact`; optional `info` is JSON-serializable data
(or `None`). `get_snn.py` saves the artifact to the requested `.pt` path and,
when info is present, saves it to a sibling file with the same stem and a
`.json` extension. 


## Artifact lifecycle

```python
artifact.reset_state()
y_t = artifact(image_step)
artifact.save(path)
loaded = artifact.load(path)
```

The checkpoint stores the converted visual model, `ConversionParams`, and
the `ConversionPlan`. The converted model contains neuron modules and their
registered threshold buffers, so loading does not require the original ANN,
calibration dataloader, or threshold cache. The ANN text tower is retained by
the caller when semantic CLIP image/text evaluation is needed.

## Temporal coding

`to_snn.coding.CODING` is an abstract base class with two methods:

- `accumulate(spikes, time_dimention=0)` encodes a complete history. A tensor
  may put time on the selected axis; a sequence always puts time on axis zero.
- `update(previous, spike, t)` returns the coded value through one-based
  timestep `t`. Start with `previous=None, t=1`, then pass each returned value
  into the next call. Implementations must not mutate inputs or retain
  sequence state on the encoder.

`RateCoding` uses `previous + (spike - previous) / t`;
`DifferentialCoding` uses `previous + spike / t`. Both return the first spike
at `t=1` with the spike's dtype. Streaming state preserves the input dtype,
including FP16 and BF16. Full-history `accumulate` retains its existing
reduction behavior. Different reduction orders and precision mean streaming
and full-history results need not be bitwise identical.

The existing names `rate_coding` and `differential_coding` are stateless
instances, and calling them directly still delegates to `accumulate`.

## Shared neuron library

`to_snn.neurons` owns the neuron implementations used by both DCGS and MMSE:

```python
from to_snn.neurons import IF, MTH

if_neuron = IF(threshold=1.0)
mth_neuron = MTH(threshold=1.0, num_thresholds=4)
```

Both classes expose single-step `forward(x)` and `reset_state()`. Conversion
and MMSE threshold search use these same classes.

## SpikeCLIP finetuned Spikingformer visual

The `openai_vit_B_16` backend supplies untouched OpenAI CLIP ViT-B/16 ANN
text weights. Convert the supported SpikeCLIP visual checkpoint once:

```bash
conda activate llava
pip install spikingjelly==0.0.0.0.14
mkdir -p weights
python -m utils.convert_spikeclip \
  /data/home/bky/SpikeCLIP/output/cub/finetune/20261116-140105/model_best.pth.tar \
  weights/spikeclip_cub.pt
```

The converter strictly checks the complete visual state dictionary. It supports
this rate-coded Spikingformer configuration: width 384, eight attention heads,
four blocks, MLP ratio four, and 512-dimensional projected output. It records
training T=8 and the neuron parameters in the artifact; users need not specify
neuron settings. Normal loading requires this repository and SpikingJelly,
without the training project's source directory. Only load trusted artifacts.

Each visual call advances one timestep and returns projected continuous features.
`rate_coding` averages those features. State persists past T=8, and
`CLIPWithSNNVisual.encode_image` resets it at each batch boundary. The visual
adapter resizes CLIP-normalized inputs to 32×32 using bilinear interpolation
with `align_corners=False`. The ordinary CLIP preprocessing remains unchanged.

For a CUB evaluation, explicitly select the dataset and data path:

```bash
python eval_snn.py --dataset cub --data ./data/cub \
  --clip openai_vit_B_16 --snn weights/spikeclip_cub.pt \
  --coding rate_coding --T 16 --batch-size 16 --workers 0 \
  --device cuda:3 --output-dir results/spikeclip_cub
```

`scripts/eval_snn.sh` changes the model configuration while retaining its existing
ImageNet dataset selection. Change its dataset arguments explicitly when using
CUB. Extended timesteps are supported; improved accuracy is not guaranteed.

### Independent real-data acceptance

The verifier imports the actual training `Vision_Encoder` only for its reference
side. Both sides use FP32 with TF32 disabled, identical inputs and prompts, and
the same untouched OpenAI text tower. It compares fixed T=8 against the loaded
artifact through the existing wrapper. The cosine tolerance is fixed in advance
at `atol=1e-4, rtol=1e-4`: this allows FP32 temporal reduction-order differences
while placing a direct bound on normalized similarity. Scaled-logit errors are
also reported. Failure returns a nonzero exit status.

```bash
python tests/verify_spikeclip.py \
  --reference-root /data/home/bky/SpikeCLIP \
  --checkpoint /data/home/bky/SpikeCLIP/output/cub/finetune/20261116-140105/model_best.pth.tar \
  --artifact weights/spikeclip_cub.pt --data ./data/cub \
  --device cuda:3 --samples 16 --output results/spikeclip_verification.json
```

The JSON records deterministic sample indices spread across CUB, prompts,
checkpoint/artifact hashes, maximum and mean cosine/logit errors, finite/shape
checks, T=1/8/16 prefix consistency, repeated-batch reset and intervening-batch
isolation, plus a smoke run of the unmodified `eval_snn.evaluate`. The smoke
subset retains all hierarchy metadata. Dataset preparation follows
[DATASETS.md](DATASETS.md); verification additionally requires the training
project's reference dependencies. `--metadata-dir` can select an existing
`FreeGrainLearning/data` directory when running in an isolated worktree.
