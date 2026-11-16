#!/bin/bash
# Phase 1 error analysis (CPU only): per-utterance tables -> summary CSVs/figures -> audit package.
#   sbatch scripts/analysis/analysis.sbatch
#SBATCH --job-name=vm-analysis
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -euo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
python scripts/analysis/build_utt_table.py --workers 32
python scripts/analysis/summarize.py
python scripts/analysis/export_audit.py --model Qwen3-ASR-1.7B --mode auto --n 5
echo "end=$(date -Is)"
