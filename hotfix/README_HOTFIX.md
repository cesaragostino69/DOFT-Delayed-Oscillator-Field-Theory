
# DOFT v12 — Hotfix Torch/Numpy mix (patch_001)

Este parche corrige el error:
`TypeError: sum() received an invalid combination of arguments ...` 
causado por usar `np.sum` sobre `self.Y` cuando `self.Y` es un tensor de PyTorch.

## Uso
1. Copia la carpeta `hotfix/` dentro de tu árbol `MODEL/` (de modo que quede `MODEL/hotfix/`).
2. Desde `MODEL/`, ejecuta:

   ```bash
   python hotfix/apply_hotfix.py
   ```

3. Reintenta tu corrida:

   ```bash
   USE_GPU=1 bash scripts/run_quick.sh runs/quick_gpu
   ```

Si ves el mensaje `[hotfix] Reemplazos aplicados: 1`, el fix se aplicó.
Si dice que no encontró la línea, probablemente ya estaba parcheado.

## Nota
Si aparecen errores similares con `np.mean`, `np.std`, `np.clip` aplicados sobre tensores,
ajusta a sus equivalentes de torch: `torch.mean`, `torch.std`, `torch.clamp`.
