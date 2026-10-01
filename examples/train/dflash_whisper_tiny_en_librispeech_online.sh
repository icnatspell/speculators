#!/usr/bin/env bash
# Generate teacher targets once, prepare them with Speculators, then train while
# recomputing only the current sample's teacher features in memory.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
SPECULATORS="${SPECULATORS:-.venv/bin/speculators}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/whisper_dflash_tiny_en_librispeech}"
STEPS="${STEPS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
EVAL_SAMPLES="${EVAL_SAMPLES:-8}"
PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-2}"
TEACHER="${TEACHER:-openai/whisper-tiny.en}"
TEACHER_REVISION="${TEACHER_REVISION:-87c7102498dcde7456f24cfd30239ca606ed9063}"
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda}"
DRAFT_DEVICE="${DRAFT_DEVICE:-$TEACHER_DEVICE}"
PREFETCH_SAMPLES="${PREFETCH_SAMPLES:-0}"
SEQ_LENGTH="${SEQ_LENGTH:-448}"

export PYTHONPATH="src:hs_connectors/src:scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

RESPONSE_FILE="$OUTPUT_DIR/generation/responses.jsonl"
AUDIO_DIR="$OUTPUT_DIR/generation/audio"
PREPARED_DIR="$OUTPUT_DIR/prepared"
TRAIN_OUTPUT="$OUTPUT_DIR/training"

echo "=== Step 1: Generate pinned greedy teacher responses ==="
"$PYTHON" scripts/generate_whisper_responses.py \
    --teacher "$TEACHER" --teacher-revision "$TEACHER_REVISION" \
    --device "$TEACHER_DEVICE" \
    --split train.clean.100 \
    --max-samples "$MAX_SAMPLES" \
    --max-new-tokens 192 \
    --shuffle-buffer 128 \
    --output-file "$RESPONSE_FILE" \
    --audio-dir "$AUDIO_DIR"

echo "=== Step 2: Prepare token IDs, masks, and token frequencies ==="
"$SPECULATORS" prepare-data \
    --model "$TEACHER" --revision "$TEACHER_REVISION" \
    --data "$RESPONSE_FILE" \
    --output "$PREPARED_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --minimum-valid-tokens 2 \
    --num-preprocessing-workers "$PREPROCESSING_WORKERS" \
    --no-skip-token-freq

echo "=== Step 3: Train the PyTorch drafter and evaluate EAL/speed ==="
"$PYTHON" scripts/train_whisper_dflash_online.py \
    --teacher "$TEACHER" --teacher-revision "$TEACHER_REVISION" \
    --train-data "$PREPARED_DIR" \
    --response-manifest "$RESPONSE_FILE.manifest.json" \
    --split train.clean.100 \
    --eval-split validation.clean \
    --device "$DRAFT_DEVICE" \
    --teacher-device "$TEACHER_DEVICE" \
    --prefetch-samples "$PREFETCH_SAMPLES" \
    --steps "$STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --max-new-tokens 192 \
    --block-size 4 \
    --max-anchors 8 \
    --target-layer-ids 0 1 2 \
    --shuffle-buffer 128 \
    --learning-rate 1e-3 \
    --warmup-steps 2 \
    --minimum-lr-ratio 0.1 \
    --checkpoint-every 100 \
    --eval-every 100 \
    --eval-samples "$EVAL_SAMPLES" \
    --eval-repetitions 2 \
    --output-dir "$TRAIN_OUTPUT"

"$PYTHON" scripts/evaluate_whisper_dflash.py \
    --checkpoint "$TRAIN_OUTPUT/best" --device "$TEACHER_DEVICE" \
    --split validation.clean --samples "$EVAL_SAMPLES" --repetitions 3 \
    --output-dir "$OUTPUT_DIR/benchmark"
