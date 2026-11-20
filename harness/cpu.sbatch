#!/bin/bash
# Small CPU steps of the harness (never on the login node):  CMD="python harness/make_splits.py" sbatch harness/cpu.sbatch
#SBATCH --job-name=vm-hcpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -euo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) cmd=${CMD:?} start=$(date -Is)"
eval "$CMD"
echo "end=$(date -Is)"
