#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-runs/quick}"
CFG="configs/config.short.json"
N_JOBS="${N_JOBS:-1}"

PY="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python)"

echo "# using python: $PY"
echo "# using config: $CFG"
echo "# n_jobs: $N_JOBS"

"$PY" -m doft.simulation.run_sim \
  --config "$CFG" \
  --out "$OUT_DIR" \
  --n-jobs "$N_JOBS"

