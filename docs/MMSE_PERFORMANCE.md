# MMSE performance improvements (2026-11-02)

Configuration: `to_snn/configs/MMSE.yaml`, MTH, four thresholds, 50 candidates,
256 time steps, rate coding. Measurements use an A100 80 GB with PyTorch 2.1.2
and Triton 2.1.0.

## Current implementation and improvement summary

MMSE now makes two streaming ANN passes without activation-cache files: first
per-layer maxima, then candidate SSE accumulation directly on GPU. The independent
`mth_const_in` operator runs all time steps inside a kernel. CUDA chunks remain
1,048,576 activations for the bundled configuration. Layers are searched in ANN
forward order on the same stream, without worker threads or activation copies.
The converted model continues to use ordinary `MTH` for image inference.

| Stage | Change | Measured effect |
| --- | --- | --- |
| 1 | Materialize candidate inputs once to enable single-step MTH fusion | 2.08x on 65,536 activations at the original chunk size |
| 2 | Enlarge CUDA chunks from 8,192 to 1,048,576 and tune layer concurrency | Fusion plus larger chunks: 53.540 s to 0.390 s on 1,048,576 activations; two workers selected |
| 3 | Fuse time steps and decoding in `mth_const_in` | Further 11.4x: 7.209 s to 0.630 s on 8,388,608 activations; tensor peak 556 to 300 MiB |
| 4 | Two streaming ANN passes | Eliminates full-dataset activation storage and activation CPU/GPU transfers; adds one ANN pass |

These are separate controlled search experiments with different sample sizes
and timing sessions. Their speedups must not be multiplied into a claimed
end-to-end conversion speedup. Full-dataset conversion has not been timed.
Candidate count, timestep count, calibration samples and coding semantics were
preserved throughout. Restart an existing conversion to load the changes;
YAML continues to use `neuron: MTH`.

## Original diagnosis and reproduction

The original diagnosis used the following pre-fix probe on idle GPU 1. Its
eager-fallback assertion no longer holds with the current implementation:

```bash
conda run --no-capture-output -n llava python tests/profile_mmse_search.py \
    --activation /tmp/snn_mmse_7hkwwskc/0/0.pt --elements 65536 --mode workers
```

The activation path is a temporary calibration cache and may disappear.
Omit `--activation` to use seeded synthetic inputs. The probe wraps MTH only
inside its own process; it does not change the converter or running job.
Tests/scripts are intentionally ignored by this repository's Git configuration.

The cached first layer has FP16 shape `[128, 32, 112, 112]` per full batch,
15 full batches and one batch of 80: 2,000 images, 802,816,000 activations.
Measurements below use a slice, with candidate thresholds derived from its
maximum, rather than evaluating all cached images.

### Original bottlenecks

1. **MMSE bypasses MTH fusion.** `search.py` broadcasts inputs using
   `h.unsqueeze(0).expand(size, -1)`, producing a zero candidate stride.
   `MTH._can_fuse` requires contiguous inputs. All 8,192 MTH calls in the
   65,536-element reproduction used eager operations, despite `fused=True`.
   A profiler probe over 16 warmed MTH steps recorded 320 computation kernel
   launches and 32 device-to-device copies. Contiguous inputs used 16 fused
   kernels. This is a confirmed layout mismatch, not a Triton compilation delay.

2. **The default activation chunk is far too small for this workload.**
   `4_194_304 // (16 * max(8, 8 * 4)) = 8,192` scalar activations per block.
   The first cached layer alone requires 98,000 blocks, each evaluating
   four candidate groups (16/16/16/2) for 256 steps: 100,352,000 MTH calls.
   The bound was written for eager threshold intermediates. Python dispatch,
   small kernels, allocations, masking and decoding are repeated per step.
   This search is linear in T, not the earlier quadratic prefix-decoding issue.

3. **Candidate completion is a misleading indication of partial progress.**
   Blocks are the outer loop and all candidate groups run inside each block.
   Completed candidates are reported only on the final activation block.
   Before then, progress explicitly reports zero completed candidates even
   while the processed-block count grows. Almost the whole layer's work must
   finish before the first candidate group is reported complete.

4. **Four concurrent layer threads provide little benefit here.** Four equal
   searches at default chunk size took 9.637 s with one worker and 9.054 s with
   four workers on the eager path. With fusion, they took 4.666 s and 16.871 s,
   respectively. Host dispatch/allocation contention is a likely explanation;
   the experiment establishes the slowdown but does not isolate GIL time from
   allocator or CUDA driver contention. This is four independent searches on
   one GPU, not a complete multi-layer conversion benchmark.

## Stage 1: enable single-step fusion and test larger chunks

65,536 real FP16 activations, 50 candidates, T=256, one search worker.
Median of three timed runs after warmup, including search I/O and final sync:

| Input/layout | Activation chunk | Seconds | Speedup |
| --- | ---: | ---: | ---: |
| Original expanded input, eager fallback | 8,192 | 2.434 | 1.00x |
| Materialize contiguous input once per candidate group/block | 8,192 | 1.169 | 2.08x |
| Same, larger chunk | 16,384 | 0.584 | 4.17x |
| Same, larger chunk | 65,536 | 0.150 | 16.20x |

All 50 MSE values matched the baseline with zero comparison tolerance in these
tests. These are small search measurements, not an end-to-end conversion
speedup guarantee. Instrumentation counts calls in both variants. Compilation
is excluded from warmed timings.

A cached 102.8 MB activation file loaded in 0.061 s in a separate warm-filesystem
probe. That is not a cold-disk benchmark, but the tiny search reproduction is
already slow without substantial disk traffic. Disk I/O cannot explain that
reproduced slowdown.

## Stage 2: select chunk size and layer concurrency

At this stage, MMSE materialized candidate inputs once per group/block on
CUDA, outside the timestep loop. This enables the existing MTH fused kernel. CPU keeps the
broadcast view. The automatic CUDA chunk budget is increased 128-fold; the
bundled configuration uses 1,048,576 scalar activations per block instead of
8,192. Chunk sizing still scales inversely with candidate-group size and
threshold count, and the CPU budget is unchanged. Explicit `chunk_size` values
still override the default. The budget is a sizing heuristic, not a hard cap
on process GPU memory; eager fallbacks and other dtypes use more memory.
The default `layer_workers` is reduced from four to two; one remains supported
for serial layer searches. The YAML search parameters and candidate batch size
are unchanged. Existing Python processes need a restart to use these changes.

On GPU 1 (A100 80 GB), 1,048,576 real FP16 activations from the first-layer
cache gave the following results, keeping all 50 candidates and T=256:

| Search | Chunk | Seconds | Peak allocated tensor MiB |
| --- | ---: | ---: | ---: |
| Original code | 8,192 | 53.540 | Not recorded |
| Fused | 65,536 | 2.672 | 34.8 |
| Fused | 262,144 | 0.681 | 139.0 |
| Fused | 1,048,576 | 0.390 | 554.0 |

Original timing is one run after a small warmup; fused timings are medians of
three after warming each shape. The largest block is about 137x faster than
the original search in this experiment, not a full-conversion speedup claim.
Memory is PyTorch peak allocated tensor memory in the isolated search process,
excluding CUDA context and the rest of an RN50 conversion. All 50 MSE values
matched the original within 1e-12 absolute/relative tolerance using float64
comparisons, and the chosen threshold was identical.

The chunk/concurrency benchmark is available below. It runs the checked-out
implementation, so reproducing this historical stage requires its single-step
search revision; current code uses temporal fusion:

```bash
conda run -n llava python tests/benchmark_mmse_search.py \
    --activation /tmp/snn_mmse_7hkwwskc/0/0.pt
```

The optional `--baseline /path/to/original/search.py` measures the original
implementation too. The earlier `profile_mmse_search.py` is a pre-fix diagnostic
probe; its eager-fallback assertion intentionally no longer holds after the fix.

For the cached 2,000-image first layer (15 files of 128 images plus 80 images),
the new block count is `15 * 49 + 31 = 766`, down from 98,000. The final
candidate-completion reports therefore read 16/50, 32/50, 48/50, 50/50 at block
766. Prior reports remain at zero completed candidates while their block count
advances. See "Reading MMSE progress" below for all field meanings.

Concurrency was rechecked with 8,388,608 real activations per search, four
searches per run, and the selected chunk size of 1,048,576. Three-run medians:

| Layer workers | Total seconds for four searches |
| ---: | ---: |
| 1 | 12.445 |
| 2 | 11.773 |
| 4 | 14.204 |

Two workers reduce time by about 5.4% relative to one and 17.1% relative to
four for this workload. This repeats the smaller experiment's ordering and is
the reason for the default of two. Independent searches use separate CUDA
streams, matching the converter. Results were checked for all searches.
Reproduce using the benchmark's `--elements 8388608 --workers-only` options.

At this stage, all 47 tests passed on GPU 2. Regression coverage included fused
kernel invocation from candidate batching, FP16/BF16/FP32 over 256 steps,
rate and differential coding, zero-threshold candidates, uneven candidate
groups and activation/file boundaries, and CUDA conversion under the new
defaults versus small explicit chunks.

## Stage 3: constant-input temporal fusion

`to_snn/neurons/mth_const_in.py` exposes the independent, inference-only
`mth_const_in` operator. Its Triton implementation is in
`to_snn/neurons/_mth_const_in_triton.py`. The original `MTH` class and its
single-step kernel are unchanged.

```python
from to_snn.neurons import mth_const_in

# x: [activation], thresholds: scalar or list/tensor of candidate thresholds
decoded = mth_const_in(
    x, thresholds, time_steps=256, num_thresholds=4, coding="rate_coding"
)
# decoded: [candidate, activation], same dtype/device as x
```

Every call starts with zero membrane and cumulative output and repeats the
same input for all time steps. It returns the final decoded values rather than
a spike trajectory, retaining no state between calls. Zero-threshold candidates
return zero, matching the MMSE search convention. `fused=False` selects the
original eager dynamics for reference comparisons.

### Implementation and numerical semantics

One GPU program handles one candidate and a tile of activations. It loads the
constant input, loops over T on the GPU, keeps membrane/cumulative/decoded
state in registers, and writes the final decoded output. There is no expanded
candidate input allocation and no per-step Python dispatch or state writeback.
MMSE still computes squared errors and FP64 reductions after decoding.

The operator preserves the input-dtype rounding of the charged membrane,
threshold comparison boundaries, spike and state updates. Decoded accumulation
uses FP32 for FP16/BF16 inputs, matching the pre-existing MMSE search. Rate
coding sums spikes then divides by T; differential coding rounds each spike/t
to the input dtype before accumulation. This intentionally follows MMSE's
existing reduction order, not the recurrence of `RateCoding.update`.

For finite nonnegative thresholds, the adjacent-mask differences in MTH select
the largest firing level: comparisons become true monotonically as threshold
levels decrease. The new kernel selects that level directly for both positive
and negative spikes. Negative/nonfinite thresholds use the original mask
difference calculation within the kernel. Low-precision state dynamics are
still simulated for every time step; this is not an approximate closed form.

CUDA FP16/BF16/FP32 with 1–8 threshold levels uses Triton. CPU, FP64, missing
Triton and larger threshold counts fall back to the original eager MTH loop.
The input is one-dimensional; a strided input is materialized once. The kernel
was tested with PyTorch 2.1.2 / Triton 2.1.0 on an A100 80 GB. Its launch uses
128 elements and four warps per program, selected by measurement.

### MMSE integration

With `neuron: MTH`, MMSE automatically invokes `mth_const_in` for each candidate
group and activation block. YAML parameters, thresholds, T, coding, chunk size,
progress semantics and the selected model's neuron class remain unchanged.
The converted model still uses ordinary `MTH` during image inference, because
hidden-layer inputs then vary over time. Existing running conversions must be
restarted to load the new implementation.

Regression tests under `tests/` cover:

- Exact decoded-output comparisons against the eager multi-step reference for
  FP16/BF16/FP32, both codings, T=1/17/256 and K=1/4/8.
- Values at and immediately around threshold boundaries, noncontiguous inputs,
  zero/negative/tiny thresholds, negative inputs, repeated calls and CPU/FP64
  fallbacks.
- Actual temporal-kernel invocation from MMSE, uneven candidate and activation
  blocks, unchanged progress events, MSE agreement and threshold selection.

`tests/benchmark_mth_const_in.py` compares complete search calls, including
cache loading, transfers, decoding, FP64 errors and reduction. Its `--baseline`
argument takes the single-step search implementation saved before this change.
The activation cache path is selectable with `--activation`. Tests and benchmark
scripts follow this repository's convention of residing in gitignored `tests/`.

### Temporal fusion measurements

A100 80 GB, FP16, K=4, T=256, 50 candidates, 8,388,608 real first-layer
activations, chunk size 1,048,576. Three timed runs after warmup, one search
at a time, with the prior single-step-fused implementation as the baseline:

| Implementation | Median seconds | Peak allocated tensor MiB |
| --- | ---: | ---: |
| Single-step MTH fusion | 7.209 | 556 |
| `mth_const_in` | 0.630 | 300 |

This is an 11.4x search speedup and about 46% lower tensor peak memory in the
isolated benchmark process. It excludes first-use compilation and is not a
full dataset-conversion timing. All 50 candidate MSEs matched the baseline
within 1e-12 absolute/relative tolerance in FP64, and the threshold choice
was identical. Four searches took 2.532 s with one worker and 2.470 s with
two, so the existing default of two workers remains.

The full GPU suite passed 52 tests; after selecting the final launch size,
the 12 constant-input and MMSE tests were rerun for the final configuration.

Run the current search benchmark with an explicit activation cache and a saved
copy of the single-step search implementation for the comparison:

```bash
conda run -n llava python tests/benchmark_mth_const_in.py \
    --activation /path/to/activation_cache.pt \
    --baseline /path/to/single_step_search.py \
    --elements 8388608
```

## Stage 4: two-pass streaming calibration

The first pass reduces pre-ReLU activations to per-layer maxima and finite-value
flags on GPU. Scalars are transferred only after the pass, rather than forcing
a CPU synchronization at every hook. Candidate thresholds are then fixed.

The second pass runs the original ANN again. Each pre-ReLU hook splits its
current activation into bounded blocks and updates a `CandidateSearch` SSE
accumulator using temporal fusion and FP64 error reduction. It does not save,
load, clone or transfer activations to CPU. Same-stream execution ensures the
search consumes inputs before inplace ReLU overwrites them. Only image transfers
and final statistics cross devices. All thresholds are selected before any
ReLU is replaced, so downstream calibration still sees ANN activations.

The old disk `search` utility remains available for reference tests and saved
cache analysis, sharing the same accumulator. Conversion never invokes it.
`layer_workers` is retained as a validated API option but has no concurrency
effect in streaming mode; no thread pool or pending layer activation queue is
created. `candidate_batch_size` and `chunk_size` still bound temporary memory.
The script no longer needs TMPDIR. Old cache directories are not automatically
removed because another process may still be using them.

The loader must be reiterable and use identical samples with deterministic
preprocessing in both passes. One-shot iterators are rejected, and sample and
per-layer activation counts are checked between passes. Count checks cannot
detect different images with identical shapes; the caller must ensure replay
consistency. The bundled deterministic ImageNet loader meets this requirement.
Nonfinite inputs are checked in both passes, including inactive layers.

Tests compare against saved-activation reference searches, including inplace
ReLUs, and forbid `torch.save`/`torch.load` during conversion. They check that the
original ANN remains unchanged, the loader can be replayed and empty inputs
fail. Complete conversion speedup has not been measured; earlier timing tables
refer to the historical search implementations, not this two-pass pipeline.

## Current streaming progress

```text
[MMSE] pass=1/2 batches=10/391 samples=1280
[MMSE] pass=2/2 batches=10/391 samples=1280
```

Pass 1 reports maximum-statistics collection. Pass 2 reports completed image
batches: every target layer and every candidate has been evaluated for those
images. With 50,000 images and batch size 128 there are 391 batches per pass;
with 2,000 images there are 16. Each pass starts its own sample counter.
Reports occur after the first batch, every ten batches and the final batch,
with GPU synchronization so reported work is complete. A batch can take much
longer in pass 2. After pass 2, each layer's chosen threshold and MSE is printed.
There is no longer a `candidates completed=0` counter during streaming conversion.

## Historical disk-search progress

MMSE first caches ANN preactivations, then searches a threshold independently
for each ReLU layer. `cached calibration batch N` counts image batches during
the first stage. During the second stage, a typical progress line is:

```text
[MMSE] layers completed=0/51 | layer=relu1 | candidates completed=0/50 | activation blocks processed=100
```

- `layers completed`: layers whose entire threshold search has finished. This
  is a layer count, not an estimate of elapsed or remaining time; layer sizes
  differ substantially.
- `layer`: the ReLU module currently being reported. With the default two
  workers, lines for different layers can interleave.
- `candidates completed`: thresholds evaluated over **all cached activations
  of this layer**. A candidate is a threshold value, not an image or a batch.
- `activation blocks processed`: blocks of flattened scalar preactivations
  processed within this layer, not image batches or time steps. During normal
  reports, each block has evaluated every candidate for all T steps. On the
  final block, the counter includes that block as each candidate group finishes;
  the candidate counter identifies which groups are complete.

The logical loop order is activation block, candidate group, then time step.
The MTH time-step loop now executes inside the fused GPU kernel. With
50 candidates and a candidate batch size of 16, each block evaluates groups
of 16, 16, 16 and 2 candidates, each for 256 steps under the bundled config.
Before the final block, every candidate still needs more input, so the
completed-candidate count stays at zero while the block count increases.
On the final block, the candidate count advances through 16, 32, 48 and 50,
all with the same final block number. At 50 the layer-completion count rises;
the converter then prints its selected threshold and minimum MSE. With parallel
search, a finished layer may wait for an earlier layer before the converter
prints its selected threshold, because results are collected in plan order.

Intermediate block reports occur approximately every five seconds at block
boundaries, with CUDA synchronization before reporting. This is not a strict
five-second refresh interval. The initial report has zero candidates and blocks;
an inactive layer can finish immediately without searching blocks.

If cache file i contains N_i scalar activations and the search chunk is C,
the total block count is `sum(ceil(N_i / C))`. Files are split independently.
This total is not currently printed. The block fraction is a better indication
of within-layer progress than the completed-candidate count. Changing chunk
size changes the counter's units, so counts from different settings cannot be
compared directly. Counts for different layers cannot be directly compared
either because activation sizes differ.

The Python conversion API accepts `layer_workers=1` for serial layer search
and an explicit `chunk_size` override. CLI `--workers` controls data loading
instead, and image `--batch-size` does not set the activation-search chunk size.
