#!/usr/bin/env bash
# EAGLE3 recipe using the shared Whisper response/data/train/eval flow.
set -euo pipefail
export SPECULATOR_TYPE=eagle3
exec bash "$(dirname "${BASH_SOURCE[0]}")/whisper_tiny_en_librispeech_online.sh"
