#!/usr/bin/env bash
# Full PyTorch workflow. Offline responses, native preparation, online features,
# isolated dev validation, then an untouched test benchmark. Never stores hidden states.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
SPECULATORS="${SPECULATORS:-.venv/bin/speculators}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/whisper_dflash_large_v2_librispeech}"
TEACHER="${TEACHER:-openai/whisper-large-v2}"
TEACHER_REVISION="${TEACHER_REVISION:-main}"
TEACHER_SHA="$("$PYTHON" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1], revision=sys.argv[2]).sha)' "$TEACHER" "$TEACHER_REVISION")"
CUDA_DEVICE_COUNT="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
if ((CUDA_DEVICE_COUNT >= 2)); then
    DEFAULT_TEACHER_DEVICE=cuda:0
    DEFAULT_DRAFT_DEVICE=cuda:1
else
    DEFAULT_TEACHER_DEVICE=cuda
    DEFAULT_DRAFT_DEVICE=cuda
fi
TEACHER_DEVICE="${TEACHER_DEVICE:-$DEFAULT_TEACHER_DEVICE}"
DRAFT_DEVICE="${DRAFT_DEVICE:-$DEFAULT_DRAFT_DEVICE}"
DEFAULT_PREFETCH=0
if [[ "$TEACHER_DEVICE" != "$DRAFT_DEVICE" ]]; then DEFAULT_PREFETCH=2; fi
PREFETCH_SAMPLES="${PREFETCH_SAMPLES:-$DEFAULT_PREFETCH}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-$BATCH_SIZE}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
EPOCHS="${EPOCHS:-3}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-8}"
AUDIO_WORKERS="${AUDIO_WORKERS:-4}"
PRECISION="${PRECISION:-auto}"
LOSS_IMPLEMENTATION="${LOSS_IMPLEMENTATION:-fused}"
DRAFT_ATTENTION="${DRAFT_ATTENTION:-sdpa}"
SEQ_LENGTH="${SEQ_LENGTH:-448}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RESUME="${RESUME:-0}"

export PYTHONPATH="src:hs_connectors/src:scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
RESPONSE_FILE="$OUTPUT_DIR/generation/responses.jsonl"
AUDIO_DIR="$OUTPUT_DIR/generation/audio"
PREPARED_DIR="$OUTPUT_DIR/prepared"
TRAIN_OUTPUT="$OUTPUT_DIR/training"
read -r -a SPLITS <<< "${TRAIN_SPLITS:-train.clean.100 train.clean.360 train.other.500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-440}"
GENERATION_ARGS=()
if [[ -n "${GENERATION_MAX_SAMPLES:-}" ]]; then
    GENERATION_ARGS+=(--max-samples "$GENERATION_MAX_SAMPLES")
fi
RESUME_ARGS=()
if [[ "$RESUME" == 1 ]]; then RESUME_ARGS=(--resume); fi

echo "=== Step 1: Generate pinned greedy teacher responses ==="
"$PYTHON" scripts/generate_whisper_responses.py \
    --teacher "$TEACHER" --teacher-revision "$TEACHER_SHA" \
    --device "$TEACHER_DEVICE" --precision "$PRECISION" \
    --batch-size "$GENERATION_BATCH_SIZE" --audio-workers "$AUDIO_WORKERS" \
    --split "${SPLITS[@]}" --max-new-tokens "$MAX_NEW_TOKENS" --shuffle-buffer "${SHUFFLE_BUFFER:-8192}" \
    --output-file "$RESPONSE_FILE" --audio-dir "$AUDIO_DIR" "${RESUME_ARGS[@]}" "${GENERATION_ARGS[@]}"

echo "=== Step 2: Prepare token rows and bind source identity ==="
if [[ "$RESUME" == 1 && -f "$PREPARED_DIR/preparation_manifest.json" ]]; then
    echo "Reusing the prepared corpus; training verifies its source identity"
else
"$SPECULATORS" prepare-data \
    --model "$TEACHER" --revision "$TEACHER_SHA" --data "$RESPONSE_FILE" \
    --output "$PREPARED_DIR" --seq-length "$SEQ_LENGTH" --minimum-valid-tokens 2 \
    --num-preprocessing-workers "$PREPROCESSING_WORKERS" --no-skip-token-freq
fi

# Derive the optimizer budget from data exposure, including accumulation.
STEPS="${STEPS:-$("$PYTHON" -c 'import math,sys; from datasets import load_from_disk; print(math.ceil(math.ceil(len(load_from_disk(sys.argv[1]))*int(sys.argv[2])/int(sys.argv[3]))/int(sys.argv[4])))' "$PREPARED_DIR" "$EPOCHS" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS")}"
WARMUP_STEPS="${WARMUP_STEPS:-$((STEPS / 100))}"
EXTRA_ARGS=()
if [[ "${COMPILE:-0}" == 1 ]]; then EXTRA_ARGS+=(--compile); fi
if [[ "${TENSORBOARD:-1}" == 1 ]]; then EXTRA_ARGS+=(--tensorboard); fi

echo "=== Step 3: Batched online features and dev-clean/dev-other validation ==="
"$PYTHON" scripts/train_whisper_dflash_online.py \
    --teacher "$TEACHER" --teacher-revision "$TEACHER_SHA" \
    --train-data "$PREPARED_DIR" --response-manifest "$RESPONSE_FILE.manifest.json" \
    --audio-root "$AUDIO_DIR" --split "${SPLITS[@]}" \
    --eval-split validation.clean validation.other \
    --device "$DRAFT_DEVICE" --teacher-device "$TEACHER_DEVICE" \
    --prefetch-samples "$PREFETCH_SAMPLES" --batch-size "$BATCH_SIZE" \
    --teacher-batch-size "$TEACHER_BATCH_SIZE" --audio-workers "$AUDIO_WORKERS" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --bucket-buffer "$((BATCH_SIZE * 16))" --precision "$PRECISION" \
    --draft-attention "$DRAFT_ATTENTION" --loss-implementation "$LOSS_IMPLEMENTATION" \
    --steps "$STEPS" --epochs "$EPOCHS" --max-new-tokens "$MAX_NEW_TOKENS" \
    --block-size 4 --max-anchors 32 --learning-rate "${LEARNING_RATE:-1e-3}" \
    --warmup-steps "$WARMUP_STEPS" --minimum-lr-ratio 0.1 \
    --checkpoint-every "${CHECKPOINT_EVERY:-1000}" --eval-every "${EVAL_EVERY:-1000}" --decode-every "${DECODE_EVERY:-5000}" \
    --eval-samples "$EVAL_SAMPLES" --log-every "${LOG_EVERY:-25}" \
    --output-dir "$TRAIN_OUTPUT" "${EXTRA_ARGS[@]}" "${RESUME_ARGS[@]}"

if [[ "$RUN_BENCHMARK" == 1 ]]; then
    echo "=== Step 4: Untouched test-clean/test-other, co-located inference ==="
    BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark}"
    if [[ -e "$BENCHMARK_OUTPUT_DIR" && "$RESUME" == 1 ]]; then
        BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR}-$(date -u +%Y%m%dT%H%M%S)"
    fi
    BENCHMARK_ARGS=()
    if [[ "${ALLOW_TOKEN_MISMATCH:-0}" == 1 ]]; then BENCHMARK_ARGS+=(--allow-token-mismatch); fi
    "$PYTHON" scripts/evaluate_whisper_dflash.py \
        --checkpoint "$TRAIN_OUTPUT/best" --device "$TEACHER_DEVICE" \
        --split test.clean test.other --repetitions 5 --precision "${BENCHMARK_PRECISION:-float32}" \
        --output-dir "$BENCHMARK_OUTPUT_DIR" "${BENCHMARK_ARGS[@]}"
fi
