#!/bin/bash
# Download the official TEST/EVAL split of one meeting corpus into data/datasets/<name>/.
#   bash scripts/data/download.sh {ami|alimeeting|chime6|dipco|notsofar}
# Only evaluation data is fetched (no training sets). Archives are deleted after extraction.
# NOTE: throughput is limited per *remote server* (OpenSLR, AMI mirror, Zenodo throttle each connection;
# HF is fast), not by JUBAIL -- big archives therefore use parallel range requests (pget). Run anywhere, e.g.
#   nohup bash scripts/data/download.sh <name> > logs/dl-<name>.log 2>&1 &
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DS=${1:?dataset name}
OUT=$ROOT/data/datasets/$DS
mkdir -p "$OUT"; cd "$OUT"
get() { wget -q -c --tries=20 --timeout=60 -O "$2" "$1"; }   # resumable, single stream
pget() { python3 "$ROOT/src/utils/pdownload.py" "$1" "$2" --workers 16; }  # parallel ranges (big archives)
untar() { tar -xf "$1" -C "$2" && rm -f "$1" "$1.done"; }

case $DS in
ami)
  # Official ASR test split ("full-corpus-asr", Kaldi/ESPnet/lhotse): 16 meetings, ~8.7 h.
  # IHM = individual headsets, SDM = Array1-01 (first mic of array 1).
  U=https://groups.inf.ed.ac.uk/ami
  for s in ES2004 IS1009 TS3003 EN2002; do for x in a b c d; do m=$s$x
    mkdir -p wav_db/$m/audio
    ( for f in Array1-01 Headset-0 Headset-1 Headset-2 Headset-3; do
      get $U/AMICorpusMirror/amicorpus/$m/audio/$m.$f.wav wav_db/$m/audio/$m.$f.wav
    done ) &
    done; wait; done
  get $U/AMICorpusAnnotations/ami_public_manual_1.6.2.zip ami_public_manual_1.6.2.zip
  ;;
alimeeting)
  # M2MeT Test set (Test_Ali_far: 8-ch array, Test_Ali_near: headsets), ~10 h, Mandarin.
  pget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz Test_Ali.tar.gz
  untar Test_Ali.tar.gz .
  ;;
chime6)
  # CHiME-6 eval sessions S01, S21 (OpenSLR 150): binaural worn mics P*.wav + Kinect arrays U0[1-6].CH[1-4].
  for f in CHiME6_eval.tar.gz CHiME6_transcriptions.tar.gz; do
    pget https://openslr.trmal.net/resources/150/$f $f; untar $f .
  done
  ;;
dipco)
  # DiPCo (Zenodo 8122551); eval sessions S01 S03 S06 S07 S08 are inside the single archive.
  pget https://zenodo.org/records/8122551/files/DipCo.tgz DipCo.tgz
  untar DipCo.tgz .
  ;;
notsofar)
  # NOTSOFAR-1 full eval set with GT (240825.1_eval_full_with_GT, 129 meetings, HF microsoft/NOTSOFAR).
  # Close-talk + single-channel (SC) devices + metadata; 7-ch MC arrays are skipped.
  . "$ROOT/env/activate.sh"
  python - <<'EOF'
from huggingface_hub import snapshot_download
p = "benchmark-datasets/eval_set/240825.1_eval_full_with_GT/MTG/*"
snapshot_download("microsoft/NOTSOFAR", repo_type="dataset", local_dir=".", max_workers=8,
                  allow_patterns=[f"{p}/*.json", f"{p}/close_talk/*.wav", f"{p}/sc_*/ch0.wav"])
EOF
  ;;
librispeech)
  # Sanity check only: LibriSpeech test-clean (OpenSLR 12), to reproduce Qwen3-ASR's published WER.
  pget https://openslr.trmal.net/resources/12/test-clean.tar.gz test-clean.tar.gz
  untar test-clean.tar.gz .
  ;;
*) echo "unknown dataset $DS"; exit 1 ;;
esac
touch "$OUT/.done_download"
echo "done $DS $(du -sh "$OUT" | cut -f1)"
