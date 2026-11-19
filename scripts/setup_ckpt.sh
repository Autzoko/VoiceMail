#!/bin/bash
# Download Qwen3-ASR (+ Qwen3-ForcedAligner, word timestamps) checkpoints into data/checkpoints/ (HF: ~10 MB/s from JUBAIL).
set -e
cd "$(dirname "$0")/.."
. env/activate.sh
for m in ${MODELS:-Qwen3-ASR-1.7B Qwen3-ASR-0.6B Qwen3-ForcedAligner-0.6B}; do
  python -c "from huggingface_hub import snapshot_download as d; d('Qwen/$m', local_dir='data/checkpoints/$m')"
done
ls -la data/checkpoints/*
