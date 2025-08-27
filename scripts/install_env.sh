#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-doft_v12}"
THIS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
ROOT_DIR="$(dirname "$THIS_DIR")"

# ---- paths to wheels (edit if stored somewhere else)
WHEELS_DIR="${WHEELS_DIR:-$HOME/wheels}"
TORCH_WHL="${TORCH_WHL:-$WHEELS_DIR/torch-2.3.0-cp310-cp310-linux_aarch64.whl}"
TV_WHL="${TV_WHL:-$WHEELS_DIR/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl}"

if [[ ! -f "$TORCH_WHL" || ! -f "$TV_WHL" ]]; then
  echo "ERROR: cannot find NVIDIA wheels:"
  echo "  $TORCH_WHL"
  echo "  $TV_WHL"
  echo "Adjust WHEELS_DIR/TORCH_WHL/TV_WHL and rerun."
  exit 1
fi

# ---- create base conda environment (Python 3.10)
conda create -y -n "$ENV_NAME" -c conda-forge python=3.10
# activate conda in this shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- NumPy/BLAS + analysis libraries (CPU)
pip install -U pip
pip install -r "$ROOT_DIR/requirements.txt"  # includes numpy<2, pandas, joblib, etc.

#pip install --no-cache-dir "typing-extensions>=4.8"
pip install --no-cache-dir --upgrade "typing-extensions>=4.8"

# now the local NVIDIA wheels:
if [ -n "$TORCH_WHL" ] && [ -f "$TORCH_WHL" ]; then
  pip install --no-cache-dir "$TORCH_WHL"
fi
if [ -n "$TV_WHL" ] && [ -f "$TV_WHL" ]; then
  pip install --no-cache-dir "$TV_WHL"
fi
# ---- packages to review --

pip install -U filelock jinja2 fsspec networkx sympy

# ---- Torch CUDA (from NVIDIA wheels, no dependencies)
pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
pip cache purge >/dev/null 2>&1 || true
pip install --no-deps --force-reinstall "$TORCH_WHL" "$TV_WHL"

# ---- activation hook: avoid overriding with ~/.local and limit BLAS threads
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/10-doft.sh" <<'EOS'
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
EOS

# ---- CUDA verification
python - <<'PY'
import torch, sys
print("python:", sys.executable)
print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available(), "| built:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA=False -> this wheel is not NVIDIA's or JetPack is missing."
PY

