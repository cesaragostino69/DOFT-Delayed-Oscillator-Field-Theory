DOFT v12b - Patch Bundle (GPU y estabilidad)

Contenido:
- patches/patch_v12b.diff  -> diff unificado (para git apply)
- scripts/apply_patch_v12b.py -> parche in-place (regex) si no usas git o el diff no aplica
- scripts/merge_runs.py -> fusiona shards en un summary global
- scripts/doctor_cuda.py -> chequeo rapido de CUDA/Torch

Opcion A (git):
  unzip doft_v12b_patch_bundle.zip -d .
  git checkout -b v12b_safety
  git apply patches/patch_v12b.diff
  pip install -r requirements.txt

Opcion B (sin git, segura):
  unzip doft_v12b_patch_bundle.zip -d .
  python scripts/apply_patch_v12b.py
  pip install -r requirements.txt

  Correr shards en paralelo:
    PAR=6 bash scripts/run_quick_multi.sh runs/quick
    python scripts/merge_runs.py runs/quick_shard* --out runs/quick_merged

Cambios clave:
- requirements.txt: agrega typing-extensions>=4.8
- utils.py: _to_numpy, ensure_numpy y spectral_entropy robusto a series cortas
- model.py: soporto torch en GPU, gamma_t consistente, suma torch/numpy, agrega lpc_ok_frac
- run_sim.py: summary.csv agrupado por (gamma, xi) con medias y recuento de replicas