#!/bin/bash
# Build the Python env on a JUBAIL compute node (login nodes cap threads and break builds):
#   sbatch -p compute -c 16 --mem 64G -t 2:00:00 -o logs/setup_env-%j.out env/setup_env.sh
# The vendored model source in src/models/qwen_asr shadows the pip `qwen-asr` package (PYTHONPATH in
# env/activate.sh); the pip package is installed only for its dependencies.
source /etc/profile 2>/dev/null || true
set -eo pipefail
cd /scratch/ll5582/VoiceMail
module load gcc/12.2.0
export CC=gcc CXX=g++ MAX_JOBS=16
. env/activate.sh
[ -x env/venv/bin/python ] || uv venv -q --python 3.12 env/venv
uv pip install "qwen-asr[vllm]" lhotse meeteval jiwer soundfile pandas zhconv click textgrid huggingface_hub
uv pip install --no-deps -e lib/chime-utils
python -c "import qwen_asr, vllm, torch, lhotse, meeteval, chime_utils; print('qwen_asr from', qwen_asr.__file__); print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
uv pip freeze > env/requirements.lock.txt
