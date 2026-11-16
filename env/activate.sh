# Source on JUBAIL:  . env/activate.sh
# Python 3.12 venv (uv) at env/venv; caches on /scratch (home quota is tiny).
_VM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH=$HOME/.local/bin:$_VM_ROOT/env/venv/bin:$PATH
export VIRTUAL_ENV=$_VM_ROOT/env/venv
export UV_CACHE_DIR=/scratch/ll5582/cache/uv UV_PYTHON_INSTALL_DIR=/scratch/ll5582/cache/uv_python
export HF_HOME=/scratch/ll5582/.cache/huggingface
[ -f /scratch/ll5582/HF_TOKEN ] && export HF_TOKEN=$(cat /scratch/ll5582/HF_TOKEN)
export PYTHONPATH=$_VM_ROOT:$_VM_ROOT/src/models:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
