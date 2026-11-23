#!/bin/bash
# Twin pilot on a few sessions: simulate -> fill missing decodes (Qwen, Audar) -> simulate ... until nothing is
# missing, then score (debug scoring: only these sessions, never official).
#   MAN=data/datasets/manifests/libricss__sdm.jsonl SPLIT=search LIMIT=3 NAME=pilot_libricss sbatch harness/twin/pilot.sbatch
# Defaults are MAN=libricss LIMIT=3: a full run must set MAN, SPLIT and LIMIT=0. DIAR=<dir> replays another
# diarization (e.g. data/outputs/harness/diar/oracle for the M1 ladder).
# Parallel cache fill (default config; each array task owns every N-th session; no scoring):
#   MAN="m1.jsonl m2.jsonl" SPLIT=search LIMIT=0 NAME=prefill_search SCORE=0 sbatch --array=0-7 harness/twin/pilot.sbatch
#SBATCH --job-name=vm-twin-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -uo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
MAN=${MAN:-data/datasets/manifests/libricss__sdm.jsonl}; SPLIT=${SPLIT:-search}; LIMIT=${LIMIT:-3}
NAME=${NAME:-pilot}; OUT=data/outputs/harness/runs/$NAME; CACHE=data/outputs/harness/cache; SCORE=${SCORE:-1}
SHARD=(); FILLSEL=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  OUT=$OUT/shard$SLURM_ARRAY_TASK_ID; SHARD=(--shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT")
  FILLSEL=(--sessions-file $OUT/_sessions.txt)
fi
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) man=$MAN split=$SPLIT limit=$LIMIT start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f harness/splits.json ] || python harness/make_splits.py
for round in $(seq 1 ${ROUNDS:-10}); do
  echo "== round $round $(date -Is)"
  python -m harness.twin.simulate --manifest $MAN --split $SPLIT --limit $LIMIT "${SHARD[@]}" ${DIAR:+--diar $DIAR} --out $OUT; rc=$?
  [ $rc -eq 0 ] && break
  [ $rc -ne 3 ] && { echo "simulate failed ($rc)"; exit $rc; }
  python -m harness.twin.fill --engine qwen --checkpoint data/checkpoints/Qwen3-ASR-1.7B --cache $CACHE \
      --audio-index data/outputs/harness/audio_index.json "${FILLSEL[@]}" || exit 1
  python -m harness.twin.fill --engine audar --checkpoint data/checkpoints/Audar-ASR-V1-Turbo --cache $CACHE \
      --audio-index data/outputs/harness/audio_index.json "${FILLSEL[@]}" || exit 1
done
[ $rc -ne 0 ] && { echo "decodes still missing after ${ROUNDS:-10} rounds"; exit 1; }
[ "$SCORE" = 1 ] && python -m harness.eval.score --hyp $OUT --ref $MAN --split $SPLIT --only-hyp --out $OUT/score
echo "end=$(date -Is)"
