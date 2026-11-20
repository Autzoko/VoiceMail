#!/bin/bash
# Separate env for NVIDIA NeMo (Nemotron-3-Diarization, the diarizer Vocali runs on Core ML). Kept apart from
# env/venv because NeMo pins its own transformers/torch stack. Build on a compute node:
#   sbatch -p nvidia --gres=gpu:1 -c 16 --mem 64G -t 2:00:00 -o logs/setup_nemo-%j.out env/setup_nemo.sh
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
uv pip install 'nemo-toolkit[asr]' soundfile pyannote.metrics pandas
python - <<'PY'
import nemo, torch
from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: F401
print("nemo", nemo.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
PY
uv pip freeze > env/requirements-nemo.lock.txt
