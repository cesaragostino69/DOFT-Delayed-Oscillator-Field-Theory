#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
export USE_GPU="${USE_GPU:-0}"
N_JOBS="${N_JOBS:-4}"
CONFIG_FILE="configs/config_phase1.json"
OUT_DIR="results/phase1_run_$(date +%Y%m%d_%H%M%S)"

# --- Environment Check ---
PY_EXE="python"
if ! command -v $PY_EXE &> /dev/null; then
    echo "Error: 'python' command not found. Please ensure Python is in your PATH."
    exit 1
fi

# --- Set PYTHONPATH (Crucial Fix) ---
# This line tells Python to look for packages inside the 'src' directory.
export PYTHONPATH="$PWD/src"

echo "# --- Starting DOFT Phase 1 Simulation ---"
echo "# Config:    $CONFIG_FILE"
echo "# Output:    $OUT_DIR"
echo "# Jobs:      $N_JOBS"
echo "# USE_GPU:   $USE_GPU"
echo "# -----------------------------------------"

# --- Run the Simulation ---
# Note the module path is now 'doft.run_sim'
$PY_EXE -m doft.run_sim \
    --config "$CONFIG_FILE" \
    --out "$OUT_DIR" \
    --n-jobs "$N_JOBS"

echo "# --- Simulation Complete ---"
echo "# Results saved in: $OUT_DIR"