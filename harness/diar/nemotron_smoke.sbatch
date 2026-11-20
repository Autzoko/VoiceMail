#!/bin/bash
# Nemotron-3-Diarization smoke test (NeMo, Vocali's `low` preset) on LibriCSS far-field sessions.
#   PRESET=low N=4 sbatch harness/diar/nemotron_smoke.sbatch
#SBATCH --job-name=vm-nemo
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -euo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
export PATH=$PWD/env/venv-nemo/bin:$PATH VIRTUAL_ENV=$PWD/env/venv-nemo
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
python harness/diar/nemotron_smoke.py --sessions ${N:-4} --preset ${PRESET:-low}
echo "end=$(date -Is)"
