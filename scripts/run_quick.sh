#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-runs/quick}"
CFG="configs/config.short.json"

PY="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python)"

echo "# using python: $PY"
echo "# using config: $CFG"
echo "# n_jobs: 1"

"$PY" -m doft.simulation.run_sim \
  --config "$CFG" \
  --out "$OUT_DIR" \
  --n-jobs 1

