# Whisper DFlash review and proposed training setup

Review date: 2026-10-01. Scope: `git diff 7883cd4...7774aa6`, comprising
`f78c168` (full LibriSpeech recipe and initial integration) and `7774aa6`
(online teacher/draft overlap). Existing unrelated untracked documents and run
directories were not modified. This is a review and proposal, not an implemented
optimization or a launched training run.

## Current additions

- Offline greedy Hugging Face Whisper response generation with pinned model and
  dataset revisions, token IDs, loss masks, source audio, references, speaker and
  chapter IDs, duration summaries, response hashes, and generation provenance.
- Native `speculators prepare-data` integration preserving Whisper metadata.
- In-memory teacher auxiliary decoder features and pre-final-LayerNorm features,
  with next-token alignment and frozen-teacher logit reconstruction checks.
- A separate Qwen3-style DFlash drafter using Whisper's vocabulary/dimensions,
  frozen embeddings/output heads, and Whisper's normalization for KL targets.
- An experimental PyTorch online trainer with AdamW, clipping, warmup/cosine
  scheduling, deterministic epoch streams, bounded latest/best checkpoints,
  optimizer/scheduler/RNG resume, and run provenance. Hidden states are never
  written as a dataset.
- Optional bounded feature production on a separate teacher GPU while the
  drafter updates on another GPU. Queue depth is not batch size.
- Cached greedy/speculative Whisper decoding, rejection rollback, EOS handling,
  sparse token suppression and block acceptance. Optional drafter context KV
  reuse; its earlier short-context measurements were slower, so it remains opt-in.
- Training loss and teacher-context EAL, free-running validation MAL/EAL,
  acceptance by position, generation-only timing, and teacher-reference WER/CER.
- Tiny smoke and full 960-hour LibriSpeech shell recipes, documentation, and
  focused correctness/resume tests. Generic vLLM serving support is outside scope.

## Standards

Parallel standards review found one documented contribution-rule violation,
four reproducibility defects, and two validation gaps/judgment calls:

1. Both commits lack `Signed-off-by` trailers required by `CONTRIBUTING.md`.
   Relevant to upstream contribution, rather than operation of this fork.
2. Resumed checkpoint/evaluation provenance copies the original invocation even
   when resumed code differs. Current resume artifacts live in `resume-*`
   directories (`scripts/train_whisper_dflash_online.py:639`), but checkpoint
   saving copies only root artifacts (`src/speculators/train/whisper_online.py:159`).
   Include the invocation/provenance chain with exported checkpoints/results.
3. Resume verifies the prepared dataset path, not content
   (`scripts/train_whisper_dflash_online.py:594`). Record preparation parameters
   and content identity, bind preparation to the response hash, and verify them.
4. Generation resume with an existing JSONL but no manifest bypasses immutable
   identity checks (`scripts/generate_whisper_responses.py:162`). The manifest is
   written only after completion. Publish an initial atomic identity manifest
   before generating; support interrupted partial-line recovery.
5. Snapshot publication temporarily removes `latest` between directory renames
   (`src/speculators/train/whisper_online.py:163`). Resume needs recovery from a
   complete previous snapshot or immutable snapshots with an atomic pointer.
6. The current local Torch is outside the declared 2.9–2.13 support range.
   Validate a clean supported environment and specify portable audio dependencies.
7. Synthetic resume tests are useful, but prepared-data identity,
   interrupted-generation, and checkpoint-publication recovery need coverage.

Good foundations include revision pins, checksums, bounded retention, state
restoration, deterministic row replay, and no hidden-state corpus. These changes
do not launch vLLM, so its `--provenance-dir` rule is not implicated.

## Spec

Against the user's requirements for fast training, high acceptance, online
features, and generation-only benchmarks, the parallel spec review found:

1. **P1: Evaluation races with feature production.** Evaluation at
   `scripts/train_whisper_dflash_online.py:570` runs inside the active producer
   lifetime. The adapter installs a model-global norm hook and expects one call
   (`src/speculators/data_generation/whisper.py:62–77`). Evaluation can trigger
   extra hook calls, corrupt extraction, and contend with benchmark GPU work.
   Pause and drain production, obtain exclusive teacher access, and resume after
   evaluation. Merely locking individual forwards does not isolate benchmark work.
2. **P1: Large-v2 recipe inherits tiny.en's revision.** Both entry points default
   to `87c7102498dcde7456f24cfd30239ca606ed9063`, while the recipe overrides only
   the model name. Resolve the requested model once and use the same SHA in every
   stage, including preparation. Explicitly configure English transcription and
   no timestamps for multilingual Large-v2.
3. **P1: Standalone benchmark is missing.** Documentation refers to
   `scripts/evaluate_whisper_dflash.py`, which is absent. Add independent checkpoint
   evaluation on untouched `test.clean` and `test.other`, with per-split reports,
   exact token agreement, WER, MAL, position acceptance, throughput/latency,
   duration/token-length buckets, and provenance.
4. **P2: Validation loss is absent.** Current WER is teacher-reference WER, and
   speculative token equality is enforced. It checks quality preservation rather
   than drafter learning. Add fixed-anchor held-out KL loss and distinguish
   teacher-context EAL from actual speculative decoding MAL.
5. **P2: Evaluation repeats unnecessary work.** Each sample gets both decoder
   warmups, repeated baseline/speculative timing, and another baseline decode just
   for text. Reuse stored teacher transcripts; separate inexpensive learning
   evaluation from occasional isolated speed benchmarks.
6. **P2: Validation sampling/reporting is weak.** Buffered shuffle/first-N selection
   does not ensure speaker coverage, and headline metrics pool clean/other. Use
   fixed seeded speaker/duration-balanced subsets and report each split. Compare
   decoder feature layers spanning depth instead of blindly using `[0,1,2]` for
   Large-v2; index zero is the embedding output.

The adapter explicitly supports one unpadded utterance. Batching requires code
changes; changing the launcher cannot provide it.

## Additional main-review findings

- **Final-token/EOS training coverage is missing in the inherited anchor helper.**
  `select_anchors` excludes the last `block_size` anchor positions
  (`src/speculators/models/dflash/utils.py:45`). For sequence length T and block
  width B, the largest anchor is T-B-1 and largest target index is T-2; final EOS
  at T-1 is never supervised. Add terminal-block coverage and masked partial
  blocks. This is inherited generic behavior, not a new regression. Before
  packing utterances, make anchors boundary-aware so no query target crosses
  documents.
- **Teacher feature extraction computes unused full-vocabulary logits.** The
  adapter calls `WhisperForConditionalGeneration`, then discards its logits.
  Call the underlying Whisper model/decoder instead and retain selected decoder
  features plus pre-final-norm state. Later, capture only selected layers rather
  than requesting every layer's hidden state.
- **The training fast path is disabled.** The teacher is loaded without an
  explicit reduced dtype; training forces eager attention/execution and eager KL
  (`src/speculators/train/whisper.py:113–117`). Add precision/backend options and
  validate AMP, native fused KL, attention backends, and compilation separately.
  Keep loss reductions numerically stable. No speed gain is established yet.
- **Training metrics and serving policy differ.** KL targets and training EAL
  use raw teacher logits; decoding applies suppression processors. Audit this
  difference, report processed top-1 agreement, and test policy-consistent targets
  or a teacher-response CE term. This is a plausible optimization, not a proven
  explanation of poor acceptance. Masking distributions must avoid NaNs in KL.
- **Audio references are absolute local URIs.** Support an explicit audio-root
  remapping or a relocatable manifest before moving prepared data to another
  machine. Keep the corpus compressed; never serialize hidden-state features.
- **Update counts do not measure comparable training exposure.** The full recipe
  caps at 200,000 single-utterance updates and one epoch. Medusa's documented
  example uses batch 8 and accumulation 2: nominally 16 utterances per update.
  Its step count cannot be copied as equivalent training exposure. Track actual
  utterances, audio hours, valid target tokens, anchors, and epochs consumed.

## Proposed implementation order

### 1. Correctness and measurement foundation

Fix teacher revision/prompt identity, producer/evaluator exclusivity, tail/EOS
coverage, generation crash/resume identity, prepared-data identity, and resume
provenance. Add the standalone evaluator and portable supported environment.
Acceptance tests: deterministic uninterrupted/resumed updates; mid-prefetch
evaluation without hook races; EOS and document boundaries; baseline/speculative
token agreement under the actual Large-v2 policy.

### 2. Throughput before scaling

1. Remove unused teacher vocabulary projection; enable a configurable precision
   path (BF16 when supported, FP16 with appropriate training scaling otherwise).
2. Batch offline response generation and teacher feature extraction with padded
   decoder masks and duration/token-length buckets. Sweep teacher microbatch
   sizes 1, 2, 4, 8 subject to memory, rather than assuming 8 will fit.
3. Combine several utterances for drafter updates. Native DFlash already supports
   packed document IDs, so evaluate feature packing after fixing boundary-aware
   anchors and resetting per-document positions. Configure anchors per document
   or target-token budgets so short clips are not overwhelmed by long ones.
4. Add accumulation for effective batches larger than the memory-limited batch.
   Accumulation improves effective batch size; it does not parallelize forwards.
5. Validate native fused KL and appropriate attention/compile options against the
   reference path. Use fixed shape buckets to control compilation overhead.
6. Prefetch CPU audio/mel work and use explicit CUDA events/streams for transfers
   where measurement supports them. Preserve bounded memory and tensor lifetimes.
7. Reduce per-update host metric transfers and file writes; publish metrics every
   10–25 updates. Preserve finite-loss/gradient checks and needed error reporting.

Measure steady-state audio-hours/hour, utterances/sec, valid targets/sec,
teacher time, queue wait, transfer/update time, peak memory, and evaluation cost.
Do not infer overlap speedups by summing phase times. Optimize the slower stage.
Retain the two-GPU producer/consumer design initially; add torchrun/DDP only if
drafter replicas would improve the measured limiting stage on the target machine.

### 3. Training and acceptance selection

- Use all three LibriSpeech training splits for the large run, with globally
  shuffled, reproducible epochs. Use native 16 kHz audio; resample only other
  sample rates. Keep the 30-second short-form limit explicit and audit exclusions,
  short responses, truncation, speaker coverage, and target-length distributions.
- Teacher responses remain offline and fixed for the pinned greedy teacher;
  teacher features remain online. Human transcripts remain WER references.
- Cache fixed dev response IDs/tokens and references, but recompute features for
  loss. Validate on dev-clean and dev-other; never use test sets to select models.
- Run short pilots on speaker-diverse data, comparing current early layers with
  depth-spanning auxiliary layers. Then compare KL against a controlled CE+KL
  option and fixed position decay against native D-Pace, if the pilot warrants it.
  Treat these as ablations with equal data exposure, not guaranteed improvements.
- Start with one draft layer and block width 4 (maximum MAL 4 including anchor).
  Compare width 8 only after useful held-out acceptance; report normalized
  acceptance and speed as well as MAL. Extra draft layers or audio cross-attention
  should require evidence that capacity/conditioning, rather than data/coverage,
  limits progress.
- Choose schedules by effective data/token exposure and learning curves, not a
  copied step budget. Test learning rate and effective batch in small pilots;
  account for optimizer/schedule changes when introducing accumulation.

Suggested metric cadence (adjust using measured cost): train loss/throughput
every 10–25 updates; fixed-anchor dev loss every 1,000; representative dev decoding
MAL every 1,000–5,000; isolated speed checks at a few selected checkpoints.
Store raw weighted loss numerators/denominators, smoothed train loss, validation
loss, position accuracies, conditional prefix acceptance, free-running MAL, and
split-specific metrics. Add local TensorBoard plots alongside existing JSONL.
Select checkpoints using declared dev MAL criteria at fixed block width; choose
architectural/block-width variants using dev generation speed and quality gates.

### 4. Final benchmark contract

- Freeze the selected checkpoint and configuration before running untouched
  LibriSpeech `test.clean` and `test.other`. Report both separately. For a
  short-form-only benchmark, disclose all >30-second exclusions and resulting
  coverage; do not describe a filtered subset as the complete test set.
- Verify teacher/speculative tokens on every tested utterance. Report both WERs
  using a pinned Whisper English normalizer and corpus edit counts. They must
  match when token equality holds; WER is not expected to improve during frozen
  teacher distillation. Existing punctuation-only normalization is insufficient
  for comparable English contractions/numbers/spelling handling.
- Use the same precision, teacher attention backend, token budget, suppression
  policy, and optimized cached baseline for both modes. Evaluate the intended
  deployment GPU placement; normally co-locate teacher/drafter on one GPU.
  Label separate-GPU inference results and resources explicitly.
- Primary speed timing starts with the first output token available and ends
  with the final token available. Exclude loading, mel preprocessing, encoder,
  decoder prompt prefill, TTFT, and startup; include all ongoing draft/verifier,
  transfer, acceptance, and cache work. Quiesce training/background workers and
  synchronize relevant devices only at timing boundaries.
- Warm up, rotate baseline/speculative ordering, save repeated per-utterance
  timings (target 3–5 repetitions), and compute speedup from sums of per-clip
  median times. For N emitted tokens use N-1 in generation throughput/TPOT.
- Report MAL, maximum MAL, accepted/proposed tokens, unconditional and conditional
  position acceptance, verifier calls, tokens/sec, TPOT, utterance generation
  p50/p95 latency, and peak memory. Include length/duration buckets and paired
  uncertainty estimates. Measure stage attribution separately from benchmarks.
- Publish model/data/GPU/software identity, checkpoint hashes, commands, patches,
  exact evaluated IDs, filtering rules, and per-sample measurements.

## Evidence gathered in this review

- Focused tests: **92 passed, 1 skipped** across Whisper features, online training,
  and response-regeneration tests. The separate-GPU test was skipped; actual
  dual-GPU throughput/correctness remains unvalidated locally.
- A controlled CPU interleaving reproduced the producer/evaluator norm-hook race
  with `Expected exactly one final decoder normalization`.
- Hugging Face model API: Large-v2 with tiny.en's revision returned HTTP 404;
  Large-v2 `main` resolved to `ae4642769ce2ad8fc292556ccea8e901f1530655`.
  This resolved SHA is review evidence, not a changed recipe default.
- An inherited-helper diagnostic with T=10, B=4 found valid anchors 2–5 and
  maximum target index 8: final token index 9 was never supervised.
- Existing local 2,000-update tiny.en run: validation MAL 1.16824, generation
  speedup 0.59409x. It predates the latest additions and is a limited dev subset;
  it does not measure the proposed Large-v2 training setup.
- A proposed real-data raw-vs-suppressed target comparison could not run because
  the pinned teacher processor was absent from the default local cache. No
  mismatch rate or acceptance-causality claim is made from that check.

## Primary references

- [LibriSpeech corpus and official train/dev/test partitions](https://www.openslr.org/12)
- [Whisper-Medusa documented training example](https://github.com/aiola-lab/whisper-medusa/blob/main/README.md)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
- [Whisper English normalization](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py)

Review totals: **7 standards findings** (one contribution-rule violation, four
reproducibility defects, two validation gaps) and **6 spec findings**. The worst
runtime issue is concurrent evaluation corrupting teacher-feature extraction.

## Implementation follow-up

Implemented batched selected-layer online features (without unused teacher logits),
CPU audio workers, length buckets, mixed precision, fused loss, SDPA, accumulation,
optional compilation, depth-spanning layers, producer/evaluation exclusion, EOS/tail
supervision, decoding-policy targets and agreement audit, optional response CE/Dpace,
fixed balanced dev loss and MAL, TensorBoard, corpus exposure, resumable manifests,
prepared-source hashes, relocatable audio, checkpoint recovery, provenance copying,
and a standalone paired generation-only test benchmark with WER and uncertainty.
The full recipe derives its budget from epochs and separates dev from untouched test.

Supported Torch 2.9 regression suite: 151 passed, three hardware skips. Local CUDA
regressions additionally pass fused/eager loss and gradients and drafter-only AMP.
Real tiny.en offline generation/preparation and eager and BF16/fused/SDPA two-update
smokes completed. A clean strict FP32 two-clip test-clean benchmark gave MAL 1.087,
0.538x generation speed and exact tokens/identical 6.67% WER. These tiny numbers
validate the pipeline only. BF16 verification can drift numerically: strict eval
rejects mismatches, while explicit opt-in reports token match rate and both WERs.
No long training run was launched; two-GPU overlap awaits target hardware validation.
