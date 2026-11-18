#!/bin/bash
# Re-run only the summary/regression step (needs the per-utterance tables from analysis.sbatch).
#   sbatch [-p preempt | -A ecomputer -p visual] scripts/analysis/summarize.sbatch
# Several copies may be queued on different partitions; the first to start cancels the pending rest.
#SBATCH --job-name=vm-summarize
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -euo pipefail
scancel --name=vm-summarize --state=PENDING --me 2>/dev/null || true
cd /scratch/ll5582/VoiceMail
. env/activate.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
python scripts/analysis/summarize.py
echo "end=$(date -Is)"
