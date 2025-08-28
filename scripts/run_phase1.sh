#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
export USE_GPU="${USE_GPU:-0}"

# --- Set PYTHONPATH for this project ---
# This line tells Python to look for packages in the 'src' directory
# so it can find the 'doft' package.
export PYTHONPATH="$PWD/src"

echo "# --- Starting DOFT Phase 1 Simulation ---"
echo "# USE_GPU:   $USE_GPU"
echo "# -----------------------------------------"

# --- Run the Simulation ---
# Now we call the module as 'doft.simulation.run_sim'
python -m doft.simulation.run_sim

echo "# --- Simulation Complete ---"
