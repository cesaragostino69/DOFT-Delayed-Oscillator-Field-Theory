#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
export USE_GPU="${USE_GPU:-0}"
N_JOBS="${N_JOBS:-4}"
CONFIG_FILE="configs/config_phase1.json"
OUT_DIR="results/phase1_run_$(date +%Y%m%d_%H%M%S)"

# --- Set PYTHONPATH for this project ---
# This line tells Python to look for packages in the 'src' directory
# so it can find the 'doft' package.
export PYTHONPATH="$PWD/src"

echo "# --- Starting DOFT Phase 1 Simulation ---"
echo "# Config:    $CONFIG_FILE"
echo "# Output:    $OUT_DIR"
echo "# Jobs:      $N_JOBS"
echo "# USE_GPU:   $USE_GPU"
echo "# -----------------------------------------"

# --- Run the Simulation ---
# Now we call the module as 'doft.simulation.run_sim'
python -m doft.simulation.run_sim \
    --config "$CONFIG_FILE" \
    --out "$OUT_DIR" \
    --n-jobs "$N_JOBS"

echo "# --- Simulation Complete ---"
echo "# Results saved to: $OUT_DIR"
