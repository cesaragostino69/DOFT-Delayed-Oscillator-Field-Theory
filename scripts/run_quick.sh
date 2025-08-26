#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-runs/quick}"
CFG="configs/config.short.json"

PY="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python)"

echo "# usando python: $PY"
echo "# usando config: $CFG"
echo "# backend:threading | tasks:27"

"$PY" -m src.run_sim \
  --config "$CFG" \
  --out "$OUT_DIR" \
  --backend threading --n-jobs 1 --log-interval 60

