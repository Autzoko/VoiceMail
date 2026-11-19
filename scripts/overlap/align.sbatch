#!/bin/bash
# Word timestamps for the close-talk manifests (Qwen3-ForcedAligner-0.6B, single forward pass, fp32 on V100).
#   sbatch scripts/overlap/align.sbatch          (MANIFESTS="..." to override; resumable)
#SBATCH --job-name=vm-align
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -uo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
export HF_HUB_OFFLINE=1
ALL="ami__ihm alimeeting__near notsofar__ihm chime6__ihm dipco__ihm"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
python scripts/overlap/align_words.py --manifests ${MANIFESTS:-$ALL} ${EXTRA:-}
echo "exit=$? end=$(date -Is)"
