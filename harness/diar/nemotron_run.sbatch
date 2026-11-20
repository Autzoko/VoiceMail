#!/bin/bash
# Nemotron-3-Diarization over the harness splits (twin input), sharded by array index:
#   MANS="data/datasets/manifests/notsofar__sdm.jsonl ..." SPLITS="search select" sbatch --array=0-3 harness/diar/nemotron_run.sbatch
#SBATCH --job-name=vm-diar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%A_%a.out
source /etc/profile 2>/dev/null || true
set -euo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
export PATH=$PWD/env/venv-nemo/bin:$PATH VIRTUAL_ENV=$PWD/env/venv-nemo
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) task=${SLURM_ARRAY_TASK_ID:-} start=$(date -Is)"
SHARD=(); [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && SHARD=(--shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT")
python harness/diar/nemotron_run.py --manifest $MANS --split ${SPLITS:-search select} --preset ${PRESET:-low} "${SHARD[@]}"
echo "end=$(date -Is)"
