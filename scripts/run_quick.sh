#!/usr/bin/env bash
set -euo pipefail

PY="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python)"

CONFIG_SRC="${DOFT_CONFIG:-configs/config_chaos.json}"
export DOFT_CONFIG="$CONFIG_SRC"
if [[ -f "$CONFIG_SRC" ]]; then
  echo "# using config file: $CONFIG_SRC"
else
  echo "# using inline config from DOFT_CONFIG variable"
fi

echo "# using python: $PY"

"$PY" -m doft.simulation.run_sim

