# EAGLE-3 for Whisper

EAGLE-3 and DFlash use the same Whisper response generation, native data
preparation, online feature extraction, training loop, checkpoints, validation,
and generation-only benchmark. The EAGLE-3 adaptation reuses
`Eagle3DraftModel`, its first-layer definitions, native rollout masks/metrics,
and KL implementations. Only Whisper alignment, verifier LayerNorm, and the
cached linear-chain proposal strategy are specific to this integration.

## Entry points

Small real-data pipeline smoke (four generated responses, two updates):

```bash
MAX_SAMPLES=4 STEPS=2 EVAL_SAMPLES=2 MAX_NEW_TOKENS=64 \
OUTPUT_DIR=runs/whisper-eagle3-smoke \
  bash examples/train/eagle3_whisper_tiny_en_librispeech_online.sh
```

The full-data recipe for a larger machine is:

```bash
bash examples/train/eagle3_whisper_large_v2_librispeech.sh
```

Both recipes delegate to the shared `whisper_*_online.sh` workflows.
`SPECULATOR_TYPE=eagle3` selects the model; the existing DFlash wrappers select
`dflash`. Python entry points are `scripts/train_whisper_online.py
--speculator-type eagle3` and the convenience wrapper
`scripts/train_whisper_eagle3_online.py`. Existing DFlash entry points remain
compatible. No large run is needed to use the smoke recipe.

Already prepared DFlash response data can be reused unchanged:

```bash
PYTHONPATH=src:hs_connectors/src:scripts .venv/bin/python \
  scripts/train_whisper_eagle3_online.py \
  --teacher openai/whisper-tiny.en \
  --teacher-revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --train-data /path/to/prepared \
  --response-manifest /path/to/generation/responses.jsonl.manifest.json \
  --audio-root /path/to/generation/audio \
  --split train.clean.100 --eval-split validation.clean \
  --device cuda --batch-size 2 --teacher-batch-size 2 \
  --steps 2 --warmup-steps 0 --max-new-tokens 64 \
  --eval-samples 2 --output-dir runs/whisper-eagle3-prepared-smoke
```

Use the exact teacher revision and source splits from the response manifest.
For a checkpoint benchmark:

```bash
PYTHONPATH=src:hs_connectors/src:scripts .venv/bin/python \
  scripts/evaluate_whisper.py \
  --checkpoint runs/whisper-eagle3-smoke/training/best \
  --device cuda --split test.clean test.other --samples 8 --repetitions 3 \
  --output-dir runs/whisper-eagle3-smoke/test-benchmark
```

Omit `--samples` for full eligible test coverage. Generation is timed from the
first available token to the final token, with preparation, encoder work, and
TTFT excluded. WER, MAL, conditional acceptance, split/bucket summaries,
precision/device metadata, checkpoint hashes, and provenance use the common
benchmark. A fixed `--sample-cache` can be reused for fair comparisons between
algorithms with the same teacher/policy/precision.

## Training alignment and objective

For teacher auxiliary features `h[t]` and decoded token `x[t]`, the first draft
step consumes `(h[t-1], x[t])`. Its target is the teacher's pre-LayerNorm state
at `t`, whose normalized logits predict `x[t+1]`. With `[prompt0, prompt1,
response0, response1, EOS]`, alignment is:

| Draft input token | Auxiliary features | Supervised next token |
| --- | --- | --- |
| prompt1 | h[0] | response0 |
| response0 | h[1] | response1 |
| response1 | h[2] | EOS |
| EOS | h[3] | masked |

Later native rollout steps feed the predicted latent state back with the next
ground-truth token, as in this repository's EAGLE-3 training. Depth defaults to
`block_size - 1`; `--ttt-step-loss-decay` controls native rollout loss decay.
The native token embeddings and output heads remain frozen. The auxiliary
projection, input/output norms, and transformer layers train. Whisper's full
vocabulary is retained and its verifier RMSNorm is replaced with a frozen copy
of the teacher's LayerNorm, including bias.

Teacher extraction is batched and uses the same selected-layer adapter as
DFlash. Native drafter rollouts run independently per utterance, avoiding
shifted inputs/targets crossing a packed document boundary. Their losses are
pooled by supervised response-token count. Vectorizing those ragged rollout
forwards is a future performance optimization. EAGLE-3 supervises all eligible
positions; `--max-anchors` is a DFlash setting and does not subsample EAGLE-3.

The optional score-only `logits_transform` hook in native EAGLE-3 applies
Whisper suppression to targets and draft scores before native loss/metrics.
It defaults to `None`, preserving existing text-model training. The same
projection supplies the raw/policy teacher-agreement audit. Optional response
CE and fused KL are supported; DFlash's Dpace position weighting is rejected
for EAGLE-3. `--raw-targets` provides the same policy-target ablation.

## Linear proposals and cache correctness

The proposer maintains native `DynamicCache` KV for verified feature/token
pairs, pre-fills the prompt before generation timing, and autoregressively
feeds predicted latent states and policy-selected tokens. It stops at EOS or
the remaining token budget. Only the final prefill query is projected through
the vocabulary head.

Speculative suffix KV is discarded after every proposal. Newly accepted tokens
are incorporated using actual teacher features on the next round. The shared
Whisper verifier performs teacher-cache rejection rollback, contiguous-prefix
acceptance, and EOS handling. EAGLE-3 CLI runs enable draft context caching by
default; `--no-cache-draft-context` gives a rebuilding reference. This is a
linear chain, with `block_size=4` meaning three proposals plus a guaranteed
teacher anchor. Tree proposals are outside this initial integration.

Checkpoints use `whisper-eagle3-experimental-v1` through the common self-contained
Whisper save/load helpers. Old DFlash checkpoints retain their format and
loader. Resume restores optimizer, scheduler, scaler, RNG, data position, and
metrics through the shared trainer. Provenance includes response/preparation
artifacts. See [the shared Whisper guide](whisper_dflash_online.md) for devices,
precision, two-GPU prefetch, data filtering, and validation/test separation.

## Smoke evidence and limits

A supported Torch 2.9 synthetic run completed two updates. Real tiny.en GPU
smokes exercised both batched teacher extraction with accumulation and the full
shell recipe (four regenerated responses, native preparation, two updates,
validation and a standalone benchmark). FP32 speculative tokens matched the
teacher and WER matched. These tiny checks establish pipeline correctness;
they do not establish useful acceptance or competitive latency. Complete local
metrics and train/eval commands, patches, hashes, and source provenance are in
`/tmp/whisper-eagle3-smoke/recipe/` and `/tmp/whisper-eagle3-smoke/real/`.
Include those artifacts when publishing the model or numerical eval results.

Regression gates cover EOS/shift alignment, native rollout updates, frozen
teacher weights, document independence including gradients, independent
checkpoint reload, cache/rebuild equivalence, rollback, token budgets, exact
verification, and accumulated-training resume for both drafter types. A CUDA
regression compares native eager loss/gradients with fused KL and SDPA.

Reduced precision inherits the shared verifier's numeric drift limitation.
Strict FP32 evaluation requires exact tokens. Reduced-precision dev evaluation
reports drift and both WERs; standalone drift reporting requires explicit
`--allow-token-mismatch`. Only one GPU is available locally; the two-GPU
integration gate remains for the larger machine.
