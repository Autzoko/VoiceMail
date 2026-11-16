#!/bin/bash
# Download Qwen3-ASR checkpoints into data/checkpoints/ (run on a JUBAIL login node; compute nodes are slow).
set -e
cd "$(dirname "$0")/.."
. env/activate.sh
for m in Qwen3-ASR-1.7B Qwen3-ASR-0.6B; do
  python -c "from huggingface_hub import snapshot_download as d; d('Qwen/$m', local_dir='data/checkpoints/$m')"
done
ls -la data/checkpoints/*
