#!/bin/bash
# DS=<dataset> sbatch scripts/data/download.sbatch   (CPU-only download job; compute nodes have internet)
#SBATCH --job-name=vm-dl
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
# NOTE: compute nodes have ~20 KB/s internet -- prefer running download.sh on a login node.
cd /scratch/ll5582/VoiceMail
echo "host=$(hostname) ds=${DS:?} start=$(date -Is)"
bash scripts/data/download.sh "$DS"
echo "end=$(date -Is) exit=$?"
