#!/usr/bin/env bash
# Full LibriSpeech recipe: generate pinned teacher targets, prepare once, then
# train the PyTorch drafter while recomputing teacher features online.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
SPECULATORS="${SPECULATORS:-.venv/bin/speculators}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/whisper_dflash_large_v2_librispeech}"
TEACHER="${TEACHER:-openai/whisper-large-v2}"
CUDA_DEVICE_COUNT="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
if ((CUDA_DEVICE_COUNT >= 2)); then
    DEFAULT_TEACHER_DEVICE="cuda:0"
    DEFAULT_DRAFT_DEVICE="cuda:1"
    DEFAULT_PREFETCH_SAMPLES=2
else
    DEFAULT_TEACHER_DEVICE="cuda"
    DEFAULT_DRAFT_DEVICE="$DEFAULT_TEACHER_DEVICE"
    DEFAULT_PREFETCH_SAMPLES=0
fi
TEACHER_DEVICE="${TEACHER_DEVICE:-$DEFAULT_TEACHER_DEVICE}"
DRAFT_DEVICE="${DRAFT_DEVICE:-$DEFAULT_DRAFT_DEVICE}"
if [[ "$TEACHER_DEVICE" == "$DRAFT_DEVICE" ]]; then
    DEFAULT_PREFETCH_SAMPLES=0
else
    DEFAULT_PREFETCH_SAMPLES=2
fi
PREFETCH_SAMPLES="${PREFETCH_SAMPLES:-$DEFAULT_PREFETCH_SAMPLES}"
STEPS="${STEPS:-200000}"
EPOCHS="${EPOCHS:-1}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-448}"

export PYTHONPATH="src:hs_connectors/src:scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

RESPONSE_FILE="$OUTPUT_DIR/generation/responses.jsonl"
AUDIO_DIR="$OUTPUT_DIR/generation/audio"
PREPARED_DIR="$OUTPUT_DIR/prepared"
TRAIN_OUTPUT="$OUTPUT_DIR/training"
SPLITS=(train.clean.100 train.clean.360 train.other.500)

echo "=== Step 1: Generate greedy Whisper Large-v2 targets ==="
"$PYTHON" scripts/generate_whisper_responses.py \
    --teacher "$TEACHER" \
    --device "$TEACHER_DEVICE" \
    --split "${SPLITS[@]}" \
    --max-new-tokens 440 \
    --shuffle-buffer 128 \
    --output-file "$RESPONSE_FILE" \
    --audio-dir "$AUDIO_DIR"

echo "=== Step 2: Prepare all 960 hours of LibriSpeech training data ==="
"$SPECULATORS" prepare-data \
    --model "$TEACHER" \
    --data "$RESPONSE_FILE" \
    --output "$PREPARED_DIR" \
    --seq-length "$SEQ_LENGTH" \
    --minimum-valid-tokens 5 \
    --num-preprocessing-workers "$PREPROCESSING_WORKERS" \
    --no-skip-token-freq

echo "=== Step 3: Train with online teacher features and dev-clean/dev-other eval ==="
"$PYTHON" scripts/train_whisper_dflash_online.py \
    --teacher "$TEACHER" \
    --train-data "$PREPARED_DIR" \
    --response-manifest "$RESPONSE_FILE.manifest.json" \
    --split "${SPLITS[@]}" \
    --eval-split validation.clean validation.other \
    --device "$DRAFT_DEVICE" \
    --teacher-device "$TEACHER_DEVICE" \
    --prefetch-samples "$PREFETCH_SAMPLES" \
    --steps "$STEPS" \
    --epochs "$EPOCHS" \
    --max-new-tokens 440 \
    --block-size 4 \
    --max-anchors 32 \
    --target-layer-ids 0 1 2 \
    --shuffle-buffer 1024 \
    --learning-rate 1e-3 \
    --warmup-steps 1000 \
    --minimum-lr-ratio 0.1 \
    --checkpoint-every 1000 \
    --eval-every 1000 \
    --eval-samples "$EVAL_SAMPLES" \
    --eval-repetitions 2 \
    --output-dir "$TRAIN_OUTPUT"
