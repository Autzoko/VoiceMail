#!/bin/bash
# CPU steps of the overlap study (no GPU). STEP selects what runs:
#   STEP=table   sbatch scripts/overlap/cpu.sbatch   real-data attribution + summary  (needs words/ from align.sbatch)
#   STEP=remix   ...                                 build the remix mixtures + manifests
#   STEP=remixan ...                                 remix attribution + summary      (needs remix ASR outputs)
#   STEP=libricss ...                                LibriCSS manifests               (needs the download)
#   STEP=libricssan ...                              LibriCSS attribution + summary
#SBATCH --job-name=vm-ovl
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -euo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${SLURM_CPUS_PER_TASK:-32}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) step=${STEP:?} start=$(date -Is)"
case $STEP in
  table)      python scripts/overlap/overlap_table.py --workers $W ${EXTRA:-}
              python scripts/overlap/overlap_summary.py --tag real ;;
  remix)      python scripts/overlap/make_remix.py ${EXTRA:-} ;;
  remixan)    python scripts/overlap/remix_table.py --workers $W
              python scripts/overlap/overlap_summary.py --tag remix --glob 'remix__*' ;;
  libricss)   python scripts/overlap/libricss.py prepare ;;
  libricssan) python scripts/overlap/libricss.py analyze --workers $W
              python scripts/overlap/overlap_summary.py --tag libricss --glob 'libricss__*' ;;
esac
echo "end=$(date -Is)"
