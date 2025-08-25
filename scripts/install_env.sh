#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-doft_v12}"
THIS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
ROOT_DIR="$(dirname "$THIS_DIR")"

# ---- rutas a wheels (editar si las guardaste en otro lado)
WHEELS_DIR="${WHEELS_DIR:-$HOME/wheels}"
TORCH_WHL="${TORCH_WHL:-$WHEELS_DIR/torch-2.3.0-cp310-cp310-linux_aarch64.whl}"
TV_WHL="${TV_WHL:-$WHEELS_DIR/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl}"

if [[ ! -f "$TORCH_WHL" || ! -f "$TV_WHL" ]]; then
  echo "ERROR: no encuentro los wheels de NVIDIA:"
  echo "  $TORCH_WHL"
  echo "  $TV_WHL"
  echo "Ajustá WHEELS_DIR/TORCH_WHL/TV_WHL y volvé a correr."
  exit 1
fi

# ---- crear env base conda (python 3.10)
conda create -y -n "$ENV_NAME" -c conda-forge python=3.10
# activar conda en este shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- libs NumPy/BLAS + análisis (CPU)
pip install -U pip
pip install -r "$ROOT_DIR/requirements.txt"  # trae numpy<2, pandas, joblib, etc.

#pip install --no-cache-dir "typing-extensions>=4.8"
pip install --no-cache-dir --upgrade "typing-extensions>=4.8"

# ahora sí, los wheels locales de NVIDIA:
if [ -n "$TORCH_WHL" ] && [ -f "$TORCH_WHL" ]; then
  pip install --no-cache-dir "$TORCH_WHL"
fi
if [ -n "$TV_WHL" ] && [ -f "$TV_WHL" ]; then
  pip install --no-cache-dir "$TV_WHL"
fi
#----new to review --

pip install -U filelock jinja2 fsspec networkx sympy

# ---- Torch CUDA (desde wheels NVIDIA, sin deps)
pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
pip cache purge >/dev/null 2>&1 || true
pip install --no-deps --force-reinstall "$TORCH_WHL" "$TV_WHL"

# ---- gancho de activación: no pisar con ~/.local y limitar hilos BLAS
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/10-doft.sh" <<'EOS'
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
EOS

# ---- verificación CUDA
python - <<'PY'
import torch, sys
print("python:", sys.executable)
print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available(), "| built:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA=False → este wheel no es el de NVIDIA o falta JetPack."
PY

