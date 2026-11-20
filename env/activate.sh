# Source on JUBAIL:  . env/activate.sh
# Python 3.12 venv (uv) at env/venv; caches on /scratch (home quota is tiny).
# meeteval / sherpa-onnx extensions need GLIBCXX_3.4.29: the system libstdc++ (CentOS 8) is too old
type module >/dev/null 2>&1 && module load gcc/12.2.0 2>/dev/null || true
_VM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH=$HOME/.local/bin:$_VM_ROOT/env/venv/bin:$PATH
export VIRTUAL_ENV=$_VM_ROOT/env/venv
export UV_CACHE_DIR=/scratch/ll5582/cache/uv UV_PYTHON_INSTALL_DIR=/scratch/ll5582/cache/uv_python
export HF_HOME=/scratch/ll5582/.cache/huggingface
[ -f /scratch/ll5582/HF_TOKEN ] && export HF_TOKEN=$(cat /scratch/ll5582/HF_TOKEN)
export PYTHONPATH=$_VM_ROOT:$_VM_ROOT/src/models:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
# login nodes cap processes+threads at 128 per user; keep BLAS/OMP pools small (override in jobs if needed)
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
