#!/bin/bash
# Qwen3-ASR oracle-segmented evaluation on the meeting test sets.
#   MODEL=Qwen3-ASR-1.7B MANIFESTS="ami__ihm ami__sdm" sbatch scripts/eval/eval.sbatch
# MODEL is a directory under data/checkpoints (downloaded by scripts/setup_ckpt.sh). Resumable: finished
# segments in data/outputs/<model>/<manifest>.jsonl are skipped on rerun.
#SBATCH --job-name=vm-eval
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -uo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
export HF_HUB_OFFLINE=1
M=${MODEL:-Qwen3-ASR-1.7B}
ALL="librispeech__clean ami__ihm ami__sdm alimeeting__near alimeeting__far notsofar__ihm notsofar__sdm chime6__ihm chime6__sdm dipco__ihm dipco__sdm"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$(git rev-parse --short HEAD) model=$M start=$(date -Is)"
python scripts/eval/run_asr.py --model data/checkpoints/$M --manifests ${MANIFESTS:-$ALL} ${EXTRA:-}
echo "exit=$? end=$(date -Is)"
