#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Set USE_GPU to 1 to attempt using a CUDA-enabled GPU.
export USE_GPU="${USE_GPU:-0}"

# Number of parallel simulation jobs to run.
N_JOBS="${N_JOBS:-4}"

# Configuration file defining the experiments.
# CORRECTED FILENAME: Removed the extra 's' from 'configs'.
CONFIG_FILE="configs/config_phase1.json"

# Directory to save all results.
OUT_DIR="results/phase1_run_$(date +%Y%m%d_%H%M%S)"

# --- Environment Check ---
PY_EXE="python"
if ! command -v $PY_EXE &> /dev/null; then
    echo "Error: 'python' command not found. Please ensure Python is in your PATH."
    exit 1
fi

if [[ "$USE_GPU" == "1" ]]; then
    echo "# Attempting to use GPU..."
    # A simple check to see if torch can see a CUDA device.
    $PY_EXE -c "import torch; exit(0) if torch.cuda.is_available() else exit(1);" || {
        echo "# WARNING: USE_GPU=1 but torch cannot find a CUDA device. Check installation."
        echo "# Continuing on CPU."
        export USE_GPU=0
    }
fi

echo "# --- Starting DOFT Phase 1 Simulation ---"
echo "# Config:    $CONFIG_FILE"
echo "# Output:    $OUT_DIR"
echo "# Jobs:      $N_JOBS"
echo "# USE_GPU:   $USE_GPU"
echo "# -----------------------------------------"

# --- Run the Simulation ---
$PY_EXE -m src.doft.run_sim \
    --config "$CONFIG_FILE" \
    --out "$OUT_DIR" \
    --n-jobs "$N_JOBS"

echo "# --- Simulation Complete ---"
echo "# Results saved in: $OUT_DIR"
