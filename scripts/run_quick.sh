#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-runs/quick_gpu}"
CFG="configs/config.short.gpu.json"

PY="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python)"

echo "# usando python: $PY"
echo "# usando config: $CFG"
echo "# backend:threading | tasks:27 | USE_GPU=${USE_GPU:-0}"

# chequeo CUDA si pediste GPU
if [[ "${USE_GPU:-0}" == "1" ]]; then
  "$PY" scripts/doctor_cuda.py
fi

"$PY" -m src.run_sim \
  --config "$CFG" \
  --out "$OUT_DIR" \
  --backend threading --n-jobs 1 --log-interval 60

