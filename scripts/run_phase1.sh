#!/usr/bin/env bash
set -euo pipefail

# --- Configuración ---
export USE_GPU="${USE_GPU:-0}"
N_JOBS="${N_JOBS:-4}"
CONFIG_FILE="configs/config_phase1.json"
OUT_DIR="results/phase1_run_$(date +%Y%m%d_%H%M%S)"

# --- Configurar PYTHONPATH (La solución para esta estructura) ---
# Esta línea le dice a Python que busque paquetes en el directorio actual.
# Como tu código está en 'src', Python encontrará el paquete 'src'.
export PYTHONPATH="$PWD"

echo "# --- Iniciando Simulación DOFT Fase 1 ---"
echo "# Config:    $CONFIG_FILE"
echo "# Salida:    $OUT_DIR"
echo "# Jobs:      $N_JOBS"
echo "# USE_GPU:   $USE_GPU"
echo "# -----------------------------------------"

# --- Ejecutar la Simulación ---
# Ahora llamamos al módulo como 'src.run_sim'
python -m src.run_sim \
    --config "$CONFIG_FILE" \
    --out "$OUT_DIR" \
    --n-jobs "$N_JOBS"

echo "# --- Simulación Completa ---"
echo "# Resultados guardados en: $OUT_DIR"


#!/usr/bin/env bash
set -euo pipefail

# --- Configuración ---
export USE_GPU="${USE_GPU:-0}"
N_JOBS="${N_JOBS:-4}"
CONFIG_FILE="configs/config_phase1.json"
OUT_DIR="results/phase1_run_$(date +%Y%m%d_%H%M%S)"

# --- Configurar PYTHONPATH (La solución para esta estructura) ---
# Esta línea le dice a Python que busque paquetes en el directorio actual.
# Como tu código está en 'src', Python encontrará el paquete 'src'.
export PYTHONPATH="$PWD"

echo "# --- Iniciando Simulación DOFT Fase 1 ---"
echo "# Config:    $CONFIG_FILE"
echo "# Salida:    $OUT_DIR"
echo "# Jobs:      $N_JOBS"
echo "# USE_GPU:   $USE_GPU"
echo "# -----------------------------------------"

# --- Ejecutar la Simulación ---
# Ahora llamamos al módulo como 'src.run_sim'
python -m src.run_sim \
    --config "$CONFIG_FILE" \
    --out "$OUT_DIR" \
    --n-jobs "$N_JOBS"

echo "# --- Simulación Completa ---"
echo "# Resultados guardados en: $OUT_DIR"