# DOFT GPU Util Hotfix v3 (minimal)

- Parchea `src/utils.py::spectral_entropy` para usar `torch.fft` en GPU si recibe tensores torch.
- Incluye `scripts/run_quick_multi.sh` para lanzar múltiples shards en paralelo.

## Uso

```bash
cd ~/MODEL
# 1) Copiar este zip y extraerlo dentro del proyecto
unzip doft_gpu_util_hotfix_v3b.zip -d .

# 2) Aplicar el parche
python hotfix/apply_hotfix_gpu_utils.py

# 3) Correr normalmente
USE_GPU=1 bash scripts/run_quick.sh runs/quick_gpu

# 4) Paralelizar si el %GPU es bajo
PAR=4 USE_GPU=1 bash scripts/run_quick_multi.sh runs/quick_gpu
```
