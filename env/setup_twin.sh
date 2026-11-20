#!/bin/bash
# Harness twin / evaluator dependencies and models (compute node, never the login node):
#   sbatch env/setup_twin.sh
#SBATCH --job-name=vm-twin-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/ll5582/VoiceMail/logs/%x-%j.out
source /etc/profile 2>/dev/null || true
set -eo pipefail
cd /scratch/ll5582/VoiceMail
. env/activate.sh
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
uv pip install -q sacrebleu pytest sherpa-onnx
# Silero VAD exactly as Vocali bundles it (ModelAssets/silero_vad.onnx, sha256 below)
VAD=data/checkpoints/silero_vad/silero_vad.onnx
EXPECT=9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6
mkdir -p "$(dirname $VAD)"
[ -f $VAD ] || curl -sSL -o $VAD https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
GOT=$(sha256sum $VAD | cut -d' ' -f1)
[ "$GOT" = "$EXPECT" ] && echo "silero_vad.onnx matches Vocali" || echo "WARNING silero_vad.onnx sha256 $GOT != Vocali $EXPECT"
# Audar-ASR-V1-Turbo (PyTorch weights; Vocali runs an MLX 8-bit conversion of the same model)
[ -f data/checkpoints/Audar-ASR-V1-Turbo/config.json ] || \
  hf download audarai/Audar-ASR-V1-Turbo --local-dir data/checkpoints/Audar-ASR-V1-Turbo --exclude "*.gguf" > /dev/null
du -sh data/checkpoints/Audar-ASR-V1-Turbo
python -c "import sherpa_onnx, sacrebleu; print('sherpa-onnx', sherpa_onnx.__version__, 'sacrebleu', sacrebleu.__version__)"
python -m pytest -q harness/eval/tests harness/twin/tests
echo "end=$(date -Is)"
