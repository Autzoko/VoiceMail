#!/bin/bash
# Separate env for NVIDIA NeMo (Nemotron-3-Diarization, the diarizer Vocali runs on Core ML). Kept apart from
# env/venv because NeMo pins its own transformers/torch stack. Build on a compute node:
#   sbatch -p nvidia --gres=gpu:1 -c 16 --mem 64G -t 6:00:00 -o logs/setup_nemo-%j.out env/setup_nemo.sh
# Use:  . env/activate.sh && export PATH=$PWD/env/venv-nemo/bin:$PATH VIRTUAL_ENV=$PWD/env/venv-nemo
source /etc/profile 2>/dev/null || true
set -eo pipefail
cd /scratch/ll5582/VoiceMail
module load gcc/12.2.0
export CC=gcc CXX=g++ MAX_JOBS=16
. env/activate.sh
export PATH=$PWD/env/venv-nemo/bin:$PATH VIRTUAL_ENV=$PWD/env/venv-nemo
[ -x env/venv-nemo/bin/python ] || uv venv -q --python 3.12 env/venv-nemo
uv pip install Cython packaging
# pin torch to the build env/venv already uses (in the uv cache): otherwise NeMo pulls a fresh cu13 stack (~2 GB),
# which did not finish downloading in 90 min (job 19470630)
echo "torch==2.9.1+cu128" > env/nemo-constraints.txt
TORCH_IDX="--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match"
uv pip install $TORCH_IDX -c env/nemo-constraints.txt torch==2.9.1+cu128
uv pip install $TORCH_IDX -c env/nemo-constraints.txt 'nemo-toolkit[asr]' soundfile pyannote.metrics pandas
# Nemotron-3-Diarization's encoder uses RoPE (self_attention_model='rope'); PyPI nemo-toolkit 3.0.0 rejects it
# (job 19475726), so NeMo comes from git main (NEMO_REF to pin a commit)
uv pip install $TORCH_IDX -c env/nemo-constraints.txt "nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${NEMO_REF:-main}"
grep -q "rope" $VIRTUAL_ENV/lib/python3.12/site-packages/nemo/collections/asr/modules/transformer_encoder.py \
  || { echo "NeMo at ${NEMO_REF:-main} still lacks RoPE in TransformerEncoder"; exit 3; }
python - <<'PY'
import nemo, torch
from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: F401
print("nemo", nemo.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
PY
uv pip freeze > env/requirements-nemo.lock.txt
