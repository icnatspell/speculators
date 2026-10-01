# Online Whisper DFlash integration

Branch: `whisper-dflash-online`.

## Current recommended workflow

Install the supported dependencies with `uv sync --extra whisper` (or
`pip install -e '.[whisper]'`). Use the custom PyTorch online trainer; Hugging
Face supplies the teacher, tokenizer, processor, and datasets, rather than
`Trainer`. The sections below this workflow also preserve earlier experiments.

Run `examples/train/dflash_whisper_large_v2_librispeech.sh` on the training
machine. The stages are offline batched teacher response generation, native
`prepare-data`, batched online teacher feature extraction and drafter training,
then standalone held-out evaluation. Teacher revision is resolved once and
pinned across stages. Responses and compressed audio are relocatable; hidden
states are never persisted. English/transcribe/no-timestamps is explicit.
Audio is resampled only when necessary to 16 kHz, and clips over 30 seconds
are excluded with coverage counts. The corpus uses all three training splits;
validation uses dev-clean/dev-other; test-clean/test-other remain untouched.

The full recipe defaults to automatic BF16/FP16 selection, FP32 master draft
weights, SDPA attention, fused KL, batch size four, accumulation two, bounded
length bucketing, and CPU audio workers. On two GPUs the teacher and drafter
use different devices with a bounded feature queue. Evaluation/checkpointing
quiesce the producer. On one GPU they run sequentially. Set batch sizes to fit
VRAM; effective utterances/update are `BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`.
The teacher microbatch is independently configurable. `COMPILE=1` is optional;
benchmark it on the target machine because variable sequence shapes can recompile.

The objective includes EOS and partial terminal blocks, prevents attention
between packed utterances, balances anchor budgets per utterance, and aligns
teacher/draft logits with suppression policy. Optional `--response-ce-weight`,
`--position-weight dpace`, and `--raw-targets` support controlled ablations.
Default feature layers span decoder depth instead of selecting only early layers.

Training reports loss, EAL proxy, throughput, audio hours and data exposure.
Fixed speaker/duration-balanced dev samples report token-weighted validation
loss every 1,000 updates and decoded MAL/acceptance by position every 5,000.
TensorBoard is enabled in the recipe. The best checkpoint maximizes pooled dev
MAL at the configured block size. WER uses Whisper English normalization and
measures transcript quality; exact speculative token agreement means its WER
must equal the teacher's; reduced-precision drift is reported explicitly. MAL, unlike training EAL, measures actual decoding.

Checkpoints save optimizer, scheduler, scaler, RNG, data position, pending
metrics, identities, and provenance. Resume replays deterministic batches;
derived Hugging Face shuffle caches do not alter the prepared corpus identity.
Use `RESUME=1` to continue. Keep generation/preparation/training provenance
alongside the artifacts when moving or publishing a run.

Standalone benchmark:

```bash
PYTHONPATH=src:hs_connectors/src:scripts .venv/bin/python scripts/evaluate_whisper_dflash.py \
  --checkpoint runs/whisper_dflash_large_v2_librispeech/training/best \
  --split test.clean test.other --device cuda --repetitions 5 \
  --output-dir runs/whisper_dflash_large_v2_librispeech/test-benchmark
```

Omit `--samples` for full eligible test coverage; specify it for a balanced pilot.
Both methods use the same teacher precision/backend, warmed caches and encoder
outputs. The full recipe defaults to FP32 for the strict reference benchmark;
set `BENCHMARK_PRECISION` to measure the intended deployment precision. Timing starts after the first token and ends at the final token;
loading, audio preparation, encoder work and TTFT are excluded. Order alternates
and repeated per-clip medians are pooled. Reports include paired bootstrap speedup
intervals, MAL, conditional acceptance, WER, per-split/length/duration results,
device metadata and checkpoint provenance. `--profile` runs separate stage
measurements outside benchmark timings. Co-locate drafter and teacher for the
primary benchmark; a separate drafter GPU is explicitly labelled.

Validation so far: supported PyTorch 2.9 CPU checks, exact accumulated-training
resume equivalence, CUDA fused/eager loss and gradient equivalence, and a short
real tiny.en run. This establishes pipeline correctness, not useful acceptance;
use a moderate learning pilot before committing substantial compute.

## Scope

Single GPU, English transcription without timestamps, streamed LibriSpeech
`train.clean.100`, frozen Whisper teacher, one-layer DFlash, block size four.
Keep only the current audio/feature batch in memory. Persist model checkpoints,
metrics and provenance, never a hidden-state corpus. No vLLM serving support is
implied by a training checkpoint.

## Implementation sequence

1. Add an in-memory Whisper feature adapter. Capture decoder auxiliary states
   and the input to its final LayerNorm. Preserve decoder-token alignment and
   mask prompt tokens and positions without a next-token target.
2. Build a separate Qwen3 draft configuration from Whisper dimensions. Load
   decoder embeddings and output projection, and use the teacher's LayerNorm
   including bias for target reconstruction. Verify reconstructed logits.
3. Add a bounded streaming loader and explicit step budgets to the trainer
   (which currently assumes indexed data and known loader lengths). Teacher
   inference runs in the main process, without GPU DataLoader workers.
4. Add a training entry point that generates teacher transcripts, extracts
   features, updates the drafter, and releases features each step. Save the
   existing train command/patch artifacts plus model and dataset revisions,
   seed, generation settings and stream position. Bound checkpoint retention.
5. Add greedy block verification with one audio encoding, causal verification,
   rejection/cache rollback, and the same token processors as the baseline.

## Smoke gates

- Tiny random Whisper and synthetic audio on CPU: feature/logit alignment,
  non-default final norm weight and bias, detached features, finite backward,
  frozen teacher, and draft parameter updates.
- `openai/whisper-tiny.en`, eight streamed training clips, ten updates: checkpoint
  reload, no hidden-state files, bounded peak RAM/VRAM and train provenance.
- Held-out development clips: identical baseline/speculative greedy tokens,
  including forced rejection and EOS handling. Record acceptance and latency.
- Learning check on 100–500 clips before larger training. Smoke completion does
  not establish useful acceptance or speedup. Benchmark the intended teacher
  checkpoint separately.

## Current progress

- Branch and plan created.
- Initial feature adapter and CPU correctness tests implemented.
- Whisper-compatible DFlash construction, finite-gradient updates and
  self-contained checkpoint save/reload implemented.
- Bounded streaming training is available through an experimental standalone
  entry point. The existing indexed-data trainer is unchanged; integration into
  its scheduler, distributed execution and resume machinery remains future work.
- Cached greedy speculative inference and a held-out acceptance/latency smoke
  harness are implemented. Decoder-cache and auxiliary-history rollback preserve
  the contiguous accepted prefix; the encoder and cross-attention cache are reused.
- Larger learning runs and supported-version validation remain next.

Initial validation: five CPU tests passed, plus Ruff lint/format checks. Run:

```bash
PYTHONPATH=src:hs_connectors/src .venv/bin/python -m pytest -q tests/unit/test_whisper_features.py
```

The local test environment reuses Torch from the sibling `dllm` environment
via a `.pth` file to avoid another large installation; it is not a portable
environment specification. Validation used Torch 2.14.0 and Transformers 5.16.1;
Torch is above the repository's supported range, so supported-version validation
is still needed before merging.

## Running the experiment

The streaming entry point additionally needs `soundfile` and `scipy`. It reads
one Parquet batch at a time with Arrow pre-buffering disabled and closes the
iterator explicitly, including on training failure. Training uses eager
attention and eager KL loss. There are no hidden-state files or loader workers.

```bash
# No teacher or audio downloads; ten synthetic CPU updates.
PYTHONPATH=src:hs_connectors/src .venv/bin/python scripts/train_whisper_dflash.py \
  --synthetic --steps 10 --output-dir /tmp/whisper-dflash-synthetic

# Stream real audio; model download is approximately 150 MB for tiny.en.
PYTHONPATH=src:hs_connectors/src .venv/bin/python scripts/train_whisper_dflash.py \
  --device cuda --steps 10 --max-samples 32 --output-dir /tmp/whisper-dflash-real
```

Use a fresh output directory. The checkpoint is experimental and must be loaded
with `load_whisper_draft`; generic HF/vLLM draft loading is not supported. The
script saves only the final model, not optimizer/resume state. Short continuations
without a complete anchored block are skipped within the sample budget.

Validation on 2026-10-01: ten unit/regression tests passed; ten synthetic updates
and ten real streamed GPU updates completed with checkpoint reload verification.
The real run consumed eleven training clips, peaked at 488,850,432 allocated CUDA
bytes, and held allocation steady at 453,082,112 bytes after optimizer setup.
Peak process RSS was approximately 3.0 GiB, including the streaming reader.
This small run does not prove memory bounds for every Parquet shard or useful
inference performance. Loss varied between clips; it is not a held-out learning
curve. No hidden-state corpus was saved.

Real smoke artifacts: `/tmp/whisper-dflash-real-20261001-memory/`, containing
checkpoint weights/config, `results.json`, `train_command.txt` and
`speculators.patch`. Results record resolved teacher/dataset revisions, checkpoint
SHA256, sample count, seed, per-step losses and GPU allocation measurements.

## Decoding smoke and training alignment correction

Real decoding comparison exposed an issue in the first training smoke: Whisper's
plain `generate()` return value removes decoder prompt tokens and EOS. Teacher
generation now requests `return_dict_in_generate=True` and uses raw `.sequences`,
with a prefix-preservation check. The earlier checkpoints above are superseded
by `/tmp/whisper-dflash-real-20261001-aligned/` (ten corrected online updates).
Checkpoint reload also explicitly restores eager attention, since HF serializes
configs without their private attention implementation setting.

```bash
PYTHONPATH=src:hs_connectors/src .venv/bin/python scripts/evaluate_whisper_dflash.py \
  --checkpoint /tmp/whisper-dflash-real-20261001-aligned \
  --output-dir /tmp/whisper-dflash-evaluation \
  --split validation.clean --samples 3 --repetitions 3 --device cuda
```

Evaluation warms both cached decoders, alternates their timing order, synchronizes
CUDA, and includes one audio encoding in each timed decode. Audio I/O and feature
processing are excluded. Every repetition must match the baseline and HF raw
generation tokens exactly. The output directory includes the eval command/patch,
copied training provenance and config, and a drafter checksum.

Smoke results on 2026-10-01, RTX 3080, three held-out clips:

- All baseline/speculative/HF token sequences match exactly.
- Four accepted draft tokens out of 204 proposals (approximately 2%).
- Sum of per-clip median times: baseline 0.557 s, speculative 1.409 s.
- Speedup 0.395x (approximately 2.5x slower). This ten-step model is a plumbing
  smoke checkpoint; it does not establish the potential of a trained drafter.
- Eighteen unit/regression tests pass, including forced acceptance/rejection,
  decoder token limits, EOS, generation prefix preservation and checkpoint reload.

Artifacts: `/tmp/whisper-dflash-eval-20261001-aligned/`.
The next experiment should train on 100–500 streamed clips and repeat held-out
evaluation before optimizing draft inference or switching to a larger teacher.

## Larger learning experiment

Completed 2026-10-01: 450 updates from 452 streamed training clips within a
500-sample budget, starting from a fresh draft with the same seed/config as the
ten-step checkpoint. No hidden-state corpus was persisted. Checkpoint reload
passed. Training took 208.7 seconds; peak allocated CUDA memory was 470.8 MiB,
with allocation fixed at 432.1 MiB after optimizer initialization. Peak process
RSS was 3.25 GiB.

```bash
PYTHONPATH=src:hs_connectors/src .venv/bin/python scripts/train_whisper_dflash.py \
  --steps 450 --max-samples 500 --device cuda \
  --output-dir /tmp/whisper-dflash-learning-500-20261001
```

Both checkpoints were evaluated on the same ten `validation.clean` clips, with
three alternating-order timing repetitions per clip and encoder time included.
All baseline/speculative/HF outputs match; baseline outputs are also identical
between the two evaluations.

| Checkpoint | Accepted / proposed | Acceptance | Speedup vs cached baseline |
| --- | --- | --- | --- |
| 10 updates | 9 / 579 | 1.55% | 0.422x |
| 450 updates | 13 / 567 | 2.29% | 0.438x |

The acceptance gain is small and no useful latency gain was demonstrated.
Separate-run timing noise prevents treating the small speedup-ratio difference
as a performance improvement. Before scaling further, use a small fixed training
set to check whether the drafter can learn high acceptance, then distinguish
optimization/data issues from limited generalization or audio conditioning.

Artifacts and complete provenance:

- Training: `/tmp/whisper-dflash-learning-500-20261001/`
- New checkpoint evaluation: `/tmp/whisper-dflash-eval-learning-500-20261001/`
- Ten-step checkpoint comparison: `/tmp/whisper-dflash-eval-tenstep-10clips-20261001/`
- Summary and logs: `/tmp/whisper-dflash-learning-comparison-20261001/`

## Fixed-set overfitting diagnostic

Completed 2026-10-01: 1,000 fresh-draft updates cycling through eight fixed clips
from one speaker, with 32 anchors per update and learning rate 0.001. Audio and
teacher-generated token IDs stay in RAM; decoder hidden states are regenerated
and discarded each update. No hidden-state corpus or audio cache is written.
Mean loss over the first/last 80 updates fell from 5.363 to 1.178. Training took
43.6 seconds and peaked at 611.7 MiB of allocated CUDA memory. Reload passed.

```bash
PYTHONPATH=src:hs_connectors/src .venv/bin/python scripts/overfit_whisper_dflash.py \
  --samples 8 --steps 1000 --output-dir /tmp/whisper-dflash-overfit8-20261001

# Explicitly permits evaluating the exact recorded training IDs.
PYTHONPATH=src:hs_connectors/src .venv/bin/python scripts/evaluate_whisper_dflash.py \
  --checkpoint /tmp/whisper-dflash-overfit8-20261001 \
  --output-dir /tmp/whisper-dflash-overfit-training-evaluation --training-set
```

Each evaluation used three timing repetitions per clip, with encoder time
included. All sixteen baseline/speculative/HF token sequences match exactly;
the evaluated training IDs exactly match the fixed set and are disjoint from the
eight `validation.clean` IDs.

| Evaluation | Accepted / proposed | Acceptance | Speedup |
| --- | --- | --- | --- |
| Eight training clips | 260 / 276 | 94.20% | 1.311x |
| Eight held-out clips | 4 / 429 | 0.93% | 0.418x |

The implementation can learn high acceptance and yield a speedup on memorized
clips. This does not demonstrate useful generalization. The earlier larger run
used only four anchors per update and far fewer visits per clip, so the results
do not isolate dataset coverage from optimization coverage. Next increase
coverage across speakers and token positions before evaluating architectural
changes such as direct audio cross-attention.

Artifacts:

- Checkpoint/provenance: `/tmp/whisper-dflash-overfit8-20261001/`
- Seen-set evaluation: `/tmp/whisper-dflash-overfit8-train-eval-20261001/`
- Held-out evaluation: `/tmp/whisper-dflash-overfit8-heldout-eval-20261001/`
- Comparison and logs: `/tmp/whisper-dflash-overfit8-comparison-20261001/`

## Full LibriSpeech training recipe

The reproducible full-data path is now in
`examples/train/dflash_whisper_large_v2_librispeech.sh`. It generates greedy
Whisper Large-v2 targets for `train.clean.100`, `train.clean.360`, and
`train.other.500`, passes the resulting JSONL through the native
`speculators prepare-data` path, and trains the drafter while recomputing only
the current batch's Whisper features. It defaults to three globally shuffled
epochs, batch size four, and two gradient accumulation steps. The update budget
is derived from prepared dataset size; set `STEPS` or `EPOCHS` to change it.
Evaluation samples are taken from both `validation.clean` and
`validation.other` at each evaluation checkpoint. The recipe is provided for a
larger training machine and has not been launched from this workspace.

Each generated row stores the teacher token IDs and mask alongside the source
audio URI, LibriSpeech reference transcript, split, speaker, chapter, and audio
duration. `prepare-data` preserves those metadata fields. The generation
manifest records per-split row counts plus audio-duration and teacher-token
length summaries, pinned model/data revisions, command, patch, and response
file hash. `--resume` continues a partially written response file without
recounting already generated rows toward each split's sample limit.

Training supports repeatable seeded shuffle passes and restores the exact row
position on resume by replaying the same epoch sequence. The prepared dataset
size times `--epochs` is the default sample budget; `--steps` remains the
optimizer-step schedule cap. This keeps hidden states out of storage while
allowing the training set to be revisited. `--max-samples` can cap that budget
for a smoke run.

Teacher feature extraction can run in a bounded producer thread while the
drafter trains on the next sample. The full-data recipe selects the first two
visible GPUs and enables a two-microbatch prefetch queue when at least two CUDA
devices are available; otherwise it uses one GPU with prefetch disabled. Set
`TEACHER_DEVICE` and `DRAFT_DEVICE` to override device selection. The queue
holds only a small number of detached feature batches in memory; queued work is
discarded and deterministically recomputed after resume. Speculative evaluation
also supports keeping the verifier and drafter on separate devices.

For a small end-to-end smoke run, use the tiny-teacher recipe and reduce its
sample/step settings. The full Large-v2 command is:

```bash
examples/train/dflash_whisper_large_v2_librispeech.sh
```

To explicitly select GPUs for concurrent teacher feature generation and draft
optimization:

```bash
TEACHER_DEVICE=cuda:0 DRAFT_DEVICE=cuda:1 PREFETCH_SAMPLES=2 \
  examples/train/dflash_whisper_large_v2_librispeech.sh
```

The output directory contains the generation manifest and command, prepared
dataset, training command/patch, checkpoint provenance, update-level train
metrics, and per-checkpoint EAL/MAL/acceptance-by-draft-position and generation
speed measurements. It also records normalized teacher-transcript WER/CER
against the LibriSpeech references to describe pseudo-label quality. Dataset
preparation and Large-v2 generation require
substantial local storage for the audio and prepared token data, but never
persist teacher hidden states.

## Latency attribution and context KV reuse

DFlash context KV reuse is available via `cache_draft_context=True` on
`speculative_whisper_decode`. It projects only the newly accepted feature suffix,
including its FC, normalization and rotary context K/V, separately for each draft
layer. Query-block K/V never persist. Twenty unit/regression tests pass, including
two-layer cached/uncached logit equivalence, suffix-only projection and rejection
handling. The original training attention path remains unchanged.

The evaluation script now compares baseline, uncached and cached drafting in
rotating order. It separately times decoder-only execution with precomputed
encoder outputs. Synchronized per-stage profiling runs are separate from the
uninstrumented speed benchmarks, because synchronizing every stage affects timing.
Those diagnostic passes attribute encoder, draft, verifier (including prefill),
and remaining token-processing/control-flow/tensor-bookkeeping time.

Measured on the same overfit checkpoint, eight clips per set, three repetitions:

| Set / mode | Total speedup | Decoder-only speedup |
| --- | --- | --- |
| Training / uncached | 1.317x | 1.290x |
| Training / cached | 1.257x | 1.211x |
| Held-out / uncached | 0.424x | 0.423x |
| Held-out / cached | 0.405x | 0.404x |

All tokens match HF/baseline exactly and cached/uncached acceptance/rejection
counts match. Context caching did not produce a speed benefit in this small,
short-context test, so it remains opt-in. Additional cache bookkeeping and GPU
launches may outweigh saved projection work here; that attribution is an
inference, not a kernel-level measurement.

The separate uncached training-set profile totals were approximately 0.056 s
encoding, 0.310 s drafting, 0.670 s verification and 1.040 s remaining work:
roughly 3%, 15%, 32% and 50%. Cached drafting took 0.406 s in its diagnostic
pass. These synchronized stage totals should not be substituted for production
latency measurements. The next low-effort target is reducing repeated token
suppression, tensor operations and host/device synchronization in the decode loop.

Artifacts with commands, patches, revisions and drafter checksum:

- Training-set profile: `/tmp/whisper-dflash-kv-profile-train-20261001/`
- Held-out profile: `/tmp/whisper-dflash-kv-profile-heldout-20261001/`
- Comparison: `/tmp/whisper-dflash-kv-profile-comparison-20261001/comparison.json`

Lint passes for new code. The existing DFlash attention file has three preexisting
`PLR0917` positional-argument warnings under the installed Ruff version; that rule
was excluded when checking this file, without changing its established signatures.

### Primary speedup: generation after the first token

The primary evaluation metric now excludes TTFT: time starts after the first
teacher-selected token is available and ends when generation finishes. Audio
loading/features, encoding, prompt prefill, and initial draft context cache
projection are outside this window. CUDA synchronizes at the two boundaries;
there are no per-stage synchronizations in benchmark passes. Both methods must
produce identical tokens. For N output tokens, throughput uses N-1 tokens and
TPOT is generation_seconds/(N-1). One-token outputs have zero measured generation
time and undefined speedup. Speedup is baseline generation time divided by
speculative generation time; aggregate using sums of per-clip medians, rather
than averaging per-clip ratios. The older encoder-excluded decoder timing still
included prompt prefill/TTFT and must not be reported as this metric.

`modes.*.generation_seconds` stores the repeated primary measurements;
`total_seconds` and `decoder_seconds` remain secondary diagnostic measurements.
Stage profiling runs separately. Draft inference, teacher verification, cache
updates and acceptance logic required during generation remain inside the primary
window, since they affect actual token throughput.

Corrected isolated GPU run on the same eight overfit training clips (three repeats,
352 tokens after the first tokens): summed per-clip median generation time was
2.7435s baseline, 2.1251s without draft context caching (1.291x), and 2.2714s with
it (1.208x). Every output matched HF. Artifacts/provenance:
`/tmp/whisper-dflash-generation-train-isolated-20261001/`.
This is the memorized training diagnostic, not a held-out speedup claim.

### Sparse suppression and block acceptance

The default decoding path prepares suppression indices once on the model device,
instead of rebuilding a vocabulary mask with `arange`/`isin` for every token.
Both ordinary greedy decoding and speculative decoding use this optimization.
For the standard Whisper suppression processors, draft candidates and verifier
predictions are selected in blocks. Candidate EOS detection takes one host
transfer per block; acceptance and accepted-EOS detection share another transfer.
Arbitrary or subclassed processors retain the sequential path so prefix-dependent
behavior is preserved. `--reference-decode` restores the original selection and
acceptance for both methods; it is intended for controlled comparisons.

Separate synchronized profile passes now expose `selection`,
`candidate_selection`, `acceptance`, and `cache_history`, as well as model stages.
Benchmark generation measurements continue to exclude preparation and TTFT.

Controlled three-repeat runs on the eight fixed training clips:

| Decode path | Baseline generation | DFlash generation | Speedup |
| --- | ---: | ---: | ---: |
| Original selection/acceptance | 2.918s | 2.374s | 1.229x |
| Sparse/block selection/acceptance | 2.349s | 1.136s | 2.068x |

Both baseline and DFlash benefit from sparse suppression. The separate profile
attributes original candidate selection + acceptance to 1.073s, versus 0.092s
optimized. In the optimized profile, generation work is approximately 58%
teacher verification, 29% draft inference and 13% decoding overhead (excluding
encoder/prefill). Acceptance remains 260/276; all outputs match HF. Draft context
caching remains opt-in: the optimized cached result is 1.978x, slower than uncached.
Run-to-run GPU variability applies; profile times include deliberate synchronization.
These speedups are on memorized training clips, not evidence of generalization.

Artifacts including eval commands, patches, training provenance and checkpoint
checksums: `/tmp/whisper-dflash-optimized-train-20261001/` and
`/tmp/whisper-dflash-reference-profile-train-20261001/`.
Comparison: `/tmp/whisper-dflash-optimization-comparison-20261001/comparison.json`.

Optimized held-out eight-clip check: all tokens match; only 4/429 proposals are
accepted. Baseline generation 0.894s versus DFlash 1.770s (0.505x). Thus the
inference overhead fix improves the high-acceptance diagnostic, while held-out
performance is still limited by the draft's lack of generalization. Artifacts:
`/tmp/whisper-dflash-optimized-heldout-20261001/`.

Validation: 33 focused unit tests pass, including sparse suppression equivalence,
reference/optimized token and counter equivalence, EOS truncation and acceptance,
custom processor fallback, caching and generation timing; Ruff and diff checks pass.

### Teacher responses, preparation, and PyTorch training

`examples/train/dflash_whisper_tiny_en_librispeech_online.sh` runs three stages:

1. `scripts/generate_whisper_responses.py` reads a pinned audio split and uses
   Hugging Face Whisper generation to create fixed greedy teacher responses.
   It stores response token IDs and audio references, with a manifest containing
   dataset/model revisions and run provenance. It does not store hidden states.
2. `speculators prepare-data` truncates and filters tokenized examples, carries
   audio references and Whisper prompt boundaries through preprocessing, and
   writes token-frequency statistics for drafter initialization.
3. `scripts/train_whisper_dflash_online.py` trains against the prepared targets
   in PyTorch. It loads one audio clip and computes that sample's Whisper encoder
   and decoder features in memory for each update; it never writes a hidden-state
   corpus to disk. It verifies that the response manifest's teacher, dataset,
   and split match the training configuration.

The example defaults to a short 10-update, 32-clip run with 8 held-out clips so
it is safe to use as a smoke test. Increase those settings deliberately for a
real experiment. Fixed held-out evaluation reports EAL/MAL and speculative
generation-only speedup; training EAL is a teacher-context proxy. Checkpoints
record the consumed data position, optimizer/scheduler and RNG state, and can be
resumed against the same prepared data and response manifest. Training and eval
provenance are stored with their outputs.

This returns to the existing PyTorch Whisper teacher and DFlash implementation.
The attempted native vLLM path was abandoned because Whisper's encoder-decoder
serving and speculative interfaces required broader changes than this training
experiment can justify. The staged response/preparation workflow retains the
useful Speculators data-generation practices without a live vLLM dependency.

### Reduced-precision correctness

FP32 remains the strict reference benchmark in the full recipe. BF16 block
verification can change an argmax relative to single-token decoding because
matrix shapes change floating-point rounding, even with causal attention.
Reduced-precision dev evaluation records token-match rates and both WERs;
FP32 dev evaluation requires exact agreement. Standalone evaluation requires
exact agreement unless `--allow-token-mismatch` is explicitly set. To measure
the intended reduced-precision deployment, use `BENCHMARK_PRECISION=auto
ALLOW_TOKEN_MISMATCH=1` and inspect mismatch rate and WER along with speed.
Do not compare FP32 timings with a reduced-precision baseline. Drafter autocast
is scoped to the drafter so it never changes teacher precision implicitly.

A two-update tiny.en smoke consumed four training clips across two passes,
with two fixed balanced dev-clean samples. Dev MAL was about 1.04. An FP32
two-clip test-clean check gave MAL 1.087 and 0.538x generation speed, with exact
tokens and identical WER (6.67%). These are pipeline checks, not learning or
performance claims. The first streaming benchmark wrote results but hit a
local native-library shutdown failure; the reusable `--sample-cache` supports
repeating inference separately from the streaming scan.

A bounded full-workflow pilot can use:

```bash
TEACHER=openai/whisper-tiny.en TRAIN_SPLITS=train.clean.100 \
GENERATION_MAX_SAMPLES=64 SHUFFLE_BUFFER=128 MAX_NEW_TOKENS=192 \
EPOCHS=3 EVAL_SAMPLES=8 EVAL_EVERY=10 DECODE_EVERY=10 LOG_EVERY=1 \
RUN_BENCHMARK=0 OUTPUT_DIR=runs/whisper-pipeline-pilot \
  examples/train/dflash_whisper_large_v2_librispeech.sh
```

For a learning pilot, increase response coverage and data passes deliberately;
measure held-out MAL rather than extrapolating from these tiny smoke results.
The GPU fused-loss/SDPA gradient regression and drafter-only autocast regression
pass locally. Only one CUDA GPU is available, so the two-GPU integration gate
remains a check for the larger machine.
