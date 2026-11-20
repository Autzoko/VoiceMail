#!/bin/bash
# L0 threshold search (GPU: missing decodes are filled as trials need them).
#   SPACE=routing TRIALS=60 NAME=l0_routing_v1 MANS="data/datasets/manifests/notsofar__sdm.jsonl ..." sbatch harness/search/l0.sbatch
#SBATCH --job-name=vm-l0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -euo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
python -c "import optuna" 2>/dev/null || uv pip install -q optuna
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) space=${SPACE:-routing} start=$(date -Is)"
python -m harness.search.l0 --manifest ${MANS:?} --space ${SPACE:-routing} --trials ${TRIALS:-60} --name ${NAME:?} \
    ${LIMIT:+--limit $LIMIT} ${COST:+--cost $COST}
echo "end=$(date -Is)"
