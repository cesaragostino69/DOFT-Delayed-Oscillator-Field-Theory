#!/usr/bin/env bash
set -euo pipefail

PY="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python)"

export DOFT_CONFIG="${DOFT_CONFIG:-configs/config_chaos.json}"

echo "# using python: $PY"
echo "# using config: $DOFT_CONFIG"

"$PY" -m doft.simulation.run_sim

