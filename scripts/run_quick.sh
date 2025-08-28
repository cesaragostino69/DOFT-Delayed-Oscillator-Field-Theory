#!/usr/bin/env bash
set -euo pipefail

PY="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python)"

echo "# using python: $PY"

"$PY" -m doft.simulation.run_sim

