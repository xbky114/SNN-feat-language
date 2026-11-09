# Transformer-to-SNN ECMT Conversion Specification

Status: design specification. This document describes the implementation to be built; it does not implement the conversion.

## Problem Statement

The project can currently convert the supported convolutional CLIP visual tower with existing conversion methods, but it cannot convert the OpenAI CLIP ViT-L/14 visual tower with the ECMT method from `Transformer-to-SNN-ECMT`.

ViT-L/14 is a transformer visual tower. Its computation includes LayerNorm, QuickGELU, Softmax, two attention matrix products, and several linear projections. ECMT converts these operations with stateful expectation-compensation modules and replaces selected linear inputs with a signed multi-threshold spiking neuron. A direct ReLU replacement does not reproduce ECMT's temporal computation.

The implementation must therefore fit ECMT into the project's ANN-to-SNN protocol: conversion receives the ANN visual tower, an explicit conversion plan, conversion parameters, and a calibration loader; conversion returns a self-contained `SNNArtifact`; one artifact call performs one timestep; and `reset_state()` starts a new static-image sequence.

## Solution

Add an ECMT conversion method for the OpenAI CLIP ViT-L/14 visual tower. The method will construct an ECMT-ready copy of the visual tower, calibrate scalar positive and negative thresholds from a caller-supplied reiterable image loader, replace the planned transformer operations with stateful ECMT modules, and return an artifact that can be saved and loaded without the ANN or calibration data.

The visual tower remains the only converted component. The CLIP text tower remains ANN code owned by the caller. Patch projection, token and positional additions, residual additions, the final class-token selection, final `ln_post`, and the final projection remain ordinary tensor operations. Transformer LayerNorm, QuickGELU, Softmax, attention matrix products, and the ECMT neuron insertion points follow the source method's temporal formulas.

The artifact exposes a tensor-in/tensor-out single-step interface. A caller repeats the same preprocessed image for each timestep and uses the existing coding interface to consume the returned per-step visual feature. ECMT state is internal to the converted visual tower and is cleared by `reset_state()`.

## User Stories

1. As a CLIP researcher, I want to request ECMT conversion through the project's generic conversion entry point, so that transformer conversion follows the same lifecycle as existing methods.
2. As a CLIP researcher, I want the conversion to accept the visual ANN tower and leave the text tower untouched, so that converted image features can be compared with the original ANN text features.
3. As a conversion caller, I want to pass a calibration loader explicitly, so that calibration uses my selected dataset and preprocessing.
4. As a conversion caller, I want loaders yielding image tensors, image/label tuples, or image dictionaries to work, so that the existing repository dataloader can be used without an adapter-specific wrapper.
5. As a conversion caller, I want calibration to use the loader's image tensors exactly as supplied, so that calibration and evaluation see identical normalized inputs.
6. As a conversion caller, I want invalid, one-shot, empty, inconsistent, or non-finite calibration input to fail with a clear error, so that a checkpoint is never produced from an invalid calibration run.
7. As a conversion caller, I want calibration to avoid retaining all ViT-L/14 activations, so that memory use remains bounded by model state and streaming statistics.
8. As a conversion caller, I want each neuron target to have a scalar positive and scalar negative threshold, so that the default matches the ECMT source method and does not multiply the checkpoint by channel or token dimensions.
9. As a conversion caller, I want positive and negative thresholds to be computed from configurable empirical tails, so that the threshold rule can be reproduced and tuned without changing conversion code.
10. As a conversion caller, I want the Softmax neuron scale to be controlled by a dedicated parameter, so that ECMT's fixed Softmax threshold behavior is preserved.
11. As a model adapter author, I want the conversion plan to enumerate ECMT's operation categories explicitly, so that conversion never silently discovers or skips a transformer operation.
12. As a model adapter author, I want the plan to identify LayerNorm, QuickGELU, Softmax, QK, AV, and neuron insertion seams, so that the method can validate that the complete visual transformer is covered.
13. As a conversion caller, I want conversion to reject a plan with missing, duplicate, unknown, or out-of-scope targets, so that partial conversion cannot look like a valid ECMT artifact.
14. As a conversion caller, I want conversion to preserve the ANN weights exactly in a deep-copied model, so that the original loaded CLIP model remains usable for text encoding and ANN comparison.
15. As a conversion caller, I want the converted model to preserve OpenAI CLIP ViT-L/14 tensor shapes and weight layout, so that existing OpenAI weights can be used without a separate checkpoint conversion.
16. As a temporal-inference caller, I want one artifact call to represent one timestep, so that the artifact composes with the project's `RateCoding` and `DifferentialCoding` implementations.
17. As a temporal-inference caller, I want ECMT modules to maintain their own state between calls, so that expectation compensation is calculated over the true timestep history.
18. As a temporal-inference caller, I want `reset_state()` to reset every ECMT neuron and compensation module, so that two independent images do not share state.
19. As a temporal-inference caller, I want a changed input shape, device, or dtype to reinitialize incompatible state safely, so that state tensors cannot be reused with the wrong activation.
20. As a temporal-inference caller, I want inference to support any positive number of timesteps, so that the checkpoint's configured default does not constrain evaluation experiments.
21. As a researcher reproducing ECMT, I want the expectation-compensation LayerNorm, QuickGELU, and Softmax modules to use the source recurrence, so that each timestep is the source method's incremental contribution.
22. As a researcher reproducing ECMT, I want the QK and AV products to use the source two-input expectation-compensation recurrence, so that time-varying queries, keys, and values are handled as ECMT defines them.
23. As a researcher reproducing ECMT, I want the signed multi-threshold neuron to expose one tensor equal to the sum of its positive and negative spikes, so that it can be inserted into ordinary PyTorch transformer operators.
24. As a researcher reproducing ECMT, I want the neuron to preserve membrane and cumulative-output dynamics, including negative retraction of prior positive output, so that the converted trajectory matches the source method.
25. As a CUDA inference caller, I want the ECMT neuron to use a fused Triton single-step kernel when supported, so that ViT-L/14 inference does not create avoidable threshold-dimension intermediates.
26. As a CPU or unsupported-CUDA caller, I want a reference PyTorch implementation to run automatically, so that correctness and conversion do not depend on Triton availability.
27. As a CUDA inference caller, I want the fused path to support FP16, BF16, and FP32 with scalar thresholds, so that it matches the project's existing reduced-precision inference paths.
28. As a checkpoint consumer, I want the artifact to include the converted model, parameters, and conversion plan, so that loading does not require the ANN, calibration loader, or threshold cache.
29. As a checkpoint consumer, I want a loaded artifact to produce the same reset trajectory as the saved artifact, so that conversion results are portable across processes.
30. As an evaluation author, I want conversion information such as selected thresholds and calibration counts to be JSON serializable, so that CLI conversion can save an inspectable sibling report.
31. As a test author, I want tiny transformer fixtures to exercise ECMT without downloading ViT-L/14 weights, so that unit tests are deterministic and inexpensive.
32. As a test author, I want the fused Triton kernel to be compared with the eager reference on supported CUDA environments, so that optimization cannot change the ECMT trajectory.
33. As a test author, I want an end-to-end smoke path using a local OpenAI CLIP ViT-L/14 checkpoint when available, so that adapter, plan, calibration, artifact, and CLIP integration are tested together.
34. As a maintainer, I want unsupported model structures and invalid parameters to fail before model mutation, so that debugging a failed conversion does not require reconstructing a partially modified model.
35. As a maintainer, I want the implementation to use factories and explicit operation categories, so that another transformer adapter can adopt ECMT without hard-coding a second model name into the conversion method.

## Implementation Decisions

### Conversion boundary and public API

- ECMT is exposed as a normal conversion method with a parameter dataclass and a `convert` entry point.
- The highest test seam is the conversion boundary: ANN visual tower + ECMT `ConversionPlan` + `ECMTConversionParams` + `calibration_loader` produces an `SNNArtifact` and optional JSON-serializable information.
- The generic conversion dispatcher remains responsible for loading the adapter and passing only the selected visual tower. ECMT owns plan validation, calibration, model copying, replacement, and artifact construction.
- The artifact uses the existing `SNNArtifact` lifecycle. It stores the converted visual model, parameters, and plan; it is evaluated in inference mode; and it supports `reset_state()`, save, and load.

### OpenAI ViT-L/14 conversion plan

- The OpenAI ViT-L/14 adapter supplies an ECMT-specific plan for its visual transformer. The plan is authoritative; ECMT does not silently scan the model and invent missing targets.
- The plan covers every transformer block and validates complete coverage of the visual tower's intended ECMT operations.
- Target categories identify the following seams:
  - signed multi-threshold neuron inputs for the FFN projections, the QKV projection, split Q/K/V tensors, post-Softmax attention, and the attention output projection;
  - expectation-compensation replacements for the visual LayerNorm modules, QuickGELU modules, and Softmax modules;
  - explicit QK and AV matrix-product seams.
- Attention is represented through explicit conversion-ready modules so that QK and AV are visible replacement seams while OpenAI CLIP's original QKV weights, output weights, heads, and tensor layout are preserved.
- The plan is validated for non-empty, unique, known target categories and complete operation coverage before any model copy is mutated.
- Compatibility with DCGS and MMSE plans for this backend is outside this specification. The ViT-L/14 plan is optimized for faithful ECMT conversion.

### Converted visual scope

- Converted: transformer LayerNorm, QuickGELU, Softmax, QK and AV products, and all ECMT neuron insertion points named by the plan.
- Kept as ordinary tensor operations: patch convolution, reshaping and permutation, class-token construction, positional addition, residual addition, final class-token selection, final `ln_post`, and final projection.
- The text tower, tokenizer, text preprocessing, and CLIP logit computation remain owned by the caller and are not passed to conversion.
- The converted model remains a visual feature producer. Pooling is not an additional conversion stage for ViT-L/14 because its class token is selected directly.

### Stateful ECMT modules

- The expectation-compensation module wraps a pure function and maintains the cumulative input and timestep count. At the first timestep it returns the function applied to the first input. At later timesteps it returns the difference between the scaled function of the new cumulative mean and the scaled function of the previous cumulative mean.
- The expectation-compensation matrix-product module maintains cumulative left input, cumulative right input, cumulative product terms, and timestep count. At timestep one it returns the direct product. At later timesteps it returns the exact two-input correction recurrence used by ECMT, including the source update order.
- Both modules return an ordinary tensor at every timestep and expose `reset_state()`.
- State is created lazily from the first input and is re-created when shape, dtype, or device changes. Reset always discards all state rather than retaining tensors from the previous sequence.

### Signed multi-threshold ECMT neuron

- The ECMT neuron lives in the shared neuron package alongside existing neuron implementations but has its own class and semantics.
- It maintains membrane potential and cumulative output. Each call integrates the current input, checks geometrically spaced positive and negative thresholds, retracts eligible prior positive output through negative spikes, updates both state tensors, and returns the sum of positive spike tensors minus negative spike tensors.
- The public output is one signed tensor, equivalent to summing the source implementation's positive/negative spike list at the call site.
- Ordinary neuron targets use calibrated scalar positive and negative scales. The number of geometric levels is independently configurable for ordinary linear targets, Q/K/V targets, and Softmax targets.
- The Softmax target uses the configured positive scale for both directions, matching the source method's Softmax handling. Other target categories use calibrated positive and negative scales.
- The geometric gap defaults to the ECMT source value and is validated as positive.
- The neuron has an eager reference path and a fused Triton path. The fused path is selected only for supported CUDA tensors, contiguous storage, supported floating dtypes, and supported level counts; otherwise the eager path runs automatically.
- No constant-input multi-step or threshold-search-specific kernel is part of ECMT. The existing constant-input optimization remains independent.
- The Triton kernel must preserve the eager operation order and dtype rounding closely enough for the defined numerical tolerance, including boundary comparisons, negative retraction, scalar threshold broadcasting, state updates, and reset behavior.

### Parameters

`ECMTConversionParams` contains the following decisions and validation rules:

- `neuron`: neuron implementation name, defaulting to `TwoSide`; the initial implementation accepts this value only and reserves the field for a future ECMT neuron variant.
- `time_steps`: positive integer defaulting to `8`, used for conversion metadata and reports; it does not cap artifact inference.
- `percent`: floating-point empirical tail parameter defaulting to `0.999`; it must lie strictly between one half and one.
- `linear_num`: positive number of geometric levels for ordinary FFN/output linear insertion points, defaulting to `8`.
- `qkv_num`: positive number of geometric levels for Q/K/V insertion points, defaulting to `8`.
- `softmax_num`: positive number of geometric levels for post-Softmax insertion points, defaulting to `8`.
- `softmax_p`: positive Softmax scale defaulting to `0.0125`, used for both positive and negative Softmax thresholds.
- `gap`: positive geometric threshold ratio defaulting to `2`, matching the ECMT source value.
- `fused`: boolean selecting the Triton fast path when available; disabling it must leave the eager reference behavior unchanged.

The parameter object is serializable through the existing YAML loader and is stored in the artifact checkpoint. Unknown configuration fields and invalid values fail before conversion starts.

### Calibration

- `calibration_loader` is required. It must be reiterable because calibration may require multiple passes over the same samples.
- A batch may be a tensor, a tuple/list whose first tensor is the image batch, or a mapping with an `image` tensor. Other forms fail clearly.
- Calibration runs in evaluation and no-grad mode on a copied ANN visual model. It does not mutate the caller's model.
- Target activations are processed as a bounded streaming statistic. No complete activation cache is retained.
- For each ordinary neuron target, calibration determines one scalar positive scale as the empirical `percent` quantile and one scalar negative scale as the negated empirical `(1 - percent)` quantile over all batch, token, head, and feature elements observed at that target. The selected scales must be finite and strictly positive for an active target; an inactive target is represented by a zero-output replacement.
- Calibration rejects empty targets, non-finite values, inconsistent sample counts across passes, and missing plan targets.
- The calibration input is the ANN input at the corresponding neuron seam. Expectation-compensation modules and matrix-product modules do not receive threshold parameters.
- Calibration uses the caller's already-preprocessed tensors. The conversion method does not download data, infer labels, or apply a second transform.
- Conversion information records the selected thresholds, target categories, sample/element counts, and parameter values in JSON-safe forms.

### Conversion and lifecycle

- Conversion deep-copies the visual tower, validates the plan and parameters, calibrates thresholds, creates the conversion-ready attention seams, and replaces targets in a deterministic plan order.
- The original ANN remains usable after conversion.
- The converted visual artifact accepts one image tensor per call and returns one visual feature tensor. The artifact does not accept a time dimension.
- External code repeats the static image, calls `reset_state()` once before a sequence, and applies the existing coding object to the per-step feature history.
- The artifact supports any positive inference timestep count. `time_steps` is metadata and a recommended evaluation value, not a runtime limit.
- `reset_state()` reaches every nested ECMT neuron and compensation module. A checkpoint loaded from disk starts with clean runtime state.
- Save/load stores the converted module graph, buffers, parameters, and plan. Loading does not load the original CLIP adapter or calibration loader.

### Errors and unsupported cases

- Fail before mutation for an invalid parameter object, unsupported target kind, incomplete plan, duplicate path, missing calibration loader, one-shot loader, empty data, non-finite calibration value, or incompatible visual structure.
- Do not silently convert a partially recognized transformer or silently skip an unsupported operation.
- Triton absence or an unsupported kernel input is a normal runtime fallback, not a conversion error. Kernel failures unrelated to unsupported inputs must remain visible rather than being swallowed.

### CLI and configuration integration

- The existing conversion CLI can select ECMT by method name and load its YAML parameter object through the generic configuration loader.
- The CLI passes the repository dataloader instance directly as `calibration_loader` and writes the artifact plus optional JSON conversion information using the existing artifact-saving convention.
- The spec does not add a second dataset format or a method-specific preprocessing pipeline.

## Testing Decisions

Tests verify externally observable behavior at the highest practical seam. They should avoid asserting private helper layout when a public module or conversion result can test the same behavior.

- Parameter tests cover valid defaults, YAML loading, invalid ranges, unknown fields, and serialization round-trip.
- Plan tests use a tiny OpenAI-style visual transformer fixture and verify that every expected block has all required ECMT target categories, that paths are unique, and that incomplete or incompatible plans fail before mutation.
- Expectation-compensation tests compare one-step and multi-step outputs with direct reference calculations, verify state accumulation, verify reset, and verify shape/dtype/device reinitialization.
- Matrix-product compensation tests cover changing left and right operands, the first-step direct product, later correction steps, reset, and numerical stability.
- Neuron tests cover positive spikes, negative retraction, multiple geometric levels, threshold boundary behavior, scalar broadcasting, state progression, reset, and changed input shape/dtype/device.
- CUDA tests compare eager and Triton neuron trajectories across FP16, BF16, and FP32, multiple level counts, positive and negative inputs, boundary-adjacent values, and reset. They are skipped when CUDA or Triton is unavailable.
- Calibration tests use small deterministic loaders in tensor, tuple, and mapping forms. They verify scalar thresholds, streaming behavior, repeated-loader requirements, non-finite rejection, missing-target rejection, and JSON-safe conversion information.
- Conversion tests use a tiny transformer to verify that the returned object is an `SNNArtifact`, that the original ANN is unchanged, that one call advances state, and that reset makes a second run reproducible.
- The first timestep is tested for tensor shape, finiteness, and state initialization only. Timestep-one closeness to the ANN is not an acceptance criterion or a required report because ECMT's signed multi-threshold quantization makes that comparison uninformative.
- Checkpoint tests save and load a converted artifact and compare reset trajectories and final features within the documented dtype tolerance.
- CLIP integration tests use a local or synthetic OpenAI-style ViT-L/14-shaped fixture. They verify visual-only conversion, ANN text tower independence, expected feature shape, and compatibility with the existing CLIP SNN wrapper and coding interface.
- A real-weight smoke command is documented and exercised when a local ViT-L/14 checkpoint and calibration data are available. Tests do not download large weights as part of the default suite.
- Existing neuron-library, coding, conversion, and inference tests remain the prior art for naming, reset behavior, CPU fallback, and checkpoint assertions.

## Out of Scope

- Converting the CLIP text tower.
- Converting the RN50 visual tower with ECMT.
- Preserving DCGS or MMSE conversion-plan behavior for the OpenAI ViT-L/14 backend.
- Designing a new temporal coding class.
- Converting patch projection, residual additions, token construction, final class-token selection, final `ln_post`, or final projection into spiking modules.
- Per-channel or per-neuron ECMT thresholds in the initial implementation.
- Training, fine-tuning, surrogate-gradient optimization, or learning thresholds after conversion.
- A constant-input-specific ECMT kernel.
- Automatic dataset download, calibration-set selection, or label-dependent calibration.
- Guaranteeing bitwise equality between ANN features and SNN features at any timestep.
- A timestep-one ANN/SNN error report or timestep-one accuracy target.
- Benchmark claims beyond the documented correctness and optional local smoke measurements.

## Further Notes

- The existing ANN/SNN protocol treats preprocessing as external. ECMT follows that rule: adapters expose preprocessing, while the calibration and evaluation loaders provide normalized image tensors.
- The source ECMT repository uses a list-valued neuron output at some call sites and immediately sums it. The tensor-valued output specified here is the equivalent boundary representation needed by this project.
- The source method has separate level counts for ordinary linear, Q/K/V, and Softmax locations. Those categories are retained so that parameter changes correspond to meaningful ECMT controls.
- The implementation should keep the eager path readable as the reference definition. Triton is an optimization of the same single-step state transition, not a second algorithm.
- The first implementation should prioritize faithful state semantics and a complete plan over broad transformer generality. New transformer adapters can later provide their own ECMT plan using the same conversion seam.
