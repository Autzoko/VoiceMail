# VoiceMail

Meeting-ASR research on top of the **Qwen3-ASR** baseline.

```
data/        datasets/ checkpoints/ outputs/ (JUBAIL only, git-ignored)   results/ (summary CSVs, versioned)
lib/         third-party code (git-ignored; scripts/setup_lib.sh)
scripts/     data/ (download, manifests)  eval/ (inference + scoring, Slurm)
src/models/  model source (vendored qwen_asr @ QwenLM/Qwen3-ASR 7c6daf7)
src/modules/ our modifications
src/utils/   text normalization, metrics, manifests, downloader
env/         activate.sh, setup_env.sh (venv lives in env/venv, git-ignored)
```

## Workflow (JUBAIL: `/scratch/ll5582/VoiceMail`)
Code is edited locally and synced with git; data, checkpoints and experiments live only on JUBAIL.

```bash
bash scripts/setup_lib.sh                       # third-party repos -> lib/
sbatch -p nvidia --gres=gpu:1 -c 16 --mem 64G -t 2:00:00 -o logs/setup_env-%j.out env/setup_env.sh
. env/activate.sh
bash scripts/setup_ckpt.sh                      # Qwen3-ASR-1.7B / 0.6B -> data/checkpoints (login node)
for d in ami alimeeting chime6 dipco notsofar; do # test splits only; login node (compute nodes ~20 KB/s)
  nohup bash scripts/data/download.sh $d > logs/dl-$d.log 2>&1 &
done
python scripts/data/prepare_manifests.py         # -> data/datasets/manifests/<ds>__<cond>.jsonl
MODEL=Qwen3-ASR-1.7B sbatch scripts/eval/eval.sbatch   # -> data/results/qwen3asr_baseline.csv
```

## Baseline evaluation protocol (v1)
Oracle-segmented utterance-level ASR on the official test/eval split of AMI, AliMeeting, NOTSOFAR-1, CHiME-6
and DiPCo, near-field (headset/close-talk) and single-distant-mic conditions. Decoding follows the official
Qwen3-ASR protocol (bf16, greedy, max_new_tokens=1024, automatic language) but with the transformers backend:
vLLM wheels need glibc>=2.31 and JUBAIL has 2.28. LibriSpeech test-clean is run first as a sanity check against
the published WER. English is scored with the
CHiME-8 Whisper-style normalizer (WER); Mandarin with punctuation-free character error rate (CER).
Details: docstrings of `scripts/data/prepare_manifests.py` and `src/utils/text_norm.py`.
