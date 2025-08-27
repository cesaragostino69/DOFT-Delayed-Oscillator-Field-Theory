#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-runs/quick}"
CFG="configs/config.short.json"

PY="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python)"

echo "# using python: $PY"
echo "# using config: $CFG"
echo "# backend: threading | tasks: 27"

"$PY" -m doft.simulation.run_sim \
  --config "$CFG" \
  --out "$OUT_DIR" \
  --backend threading --n-jobs 1 --log-interval 60

