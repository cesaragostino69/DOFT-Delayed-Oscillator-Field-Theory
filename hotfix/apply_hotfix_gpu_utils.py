#!/usr/bin/env python3
import os, re, sys, pathlib
BANNER = "[hotfix]"

def _build_new_fn():
    return '''
def spectral_entropy(x, eps: float = 1e-12):
    """
    Entropía espectral de una serie 1D.
    - Si x es torch.Tensor: usa torch.fft en el mismo device (GPU si aplica).
    - Si x es np.ndarray / lista: usa numpy.fft.
    La base del log es natural; el cambio de base es un factor constante.
    """
    try:
        import torch
    except Exception:
        torch = None
    import numpy as _np

    # --- torch path ---
    if torch is not None and isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return 0.0
        x = x - torch.mean(x)
        X = torch.fft.rfft(x)
        S = (torch.abs(X) ** 2)
        Z = torch.clamp(torch.sum(S), min=eps)
        p = S / Z
        H = -torch.sum(p * torch.log(torch.clamp(p, min=eps)))
        return H.item() if x.device.type != "cpu" else float(H)

    # --- numpy path ---
    x = _np.asarray(x)
    if x.size == 0:
        return 0.0
    X = _np.fft.rfft(x - _np.mean(x))
    S = _np.abs(X) ** 2
    Z = max(S.sum(), eps)
    p = S / Z
    H = -(p * _np.log(_np.clip(p, eps, None))).sum()
    return float(H)
'''

def patch_utils(path):
    txt = pathlib.Path(path).read_text(encoding='utf-8')
    pat = re.compile(r"def\s+spectral_entropy\s*\([^)]*\)\s*:\s*(?:\n[ \t].*)*", re.DOTALL)
    new_fn = _build_new_fn()
    if pat.search(txt):
        txt2 = pat.sub(new_fn, txt)
        pathlib.Path(path).write_text(txt2, encoding='utf-8')
        print(f"{BANNER} utils.py -> spectral_entropy patched")
        return True
    else:
        with open(path, 'a', encoding='utf-8') as f:
            f.write('\n\n' + new_fn + '\n')
        print(f"{BANNER} utils.py -> spectral_entropy appended (not found to replace)")
        return True

def main():
    here = pathlib.Path.cwd()
    candidate = here / 'src' / 'utils.py'
    if not candidate.exists():
        for p in here.parents:
            c = p / 'src' / 'utils.py'
            if c.exists():
                candidate = c
                break
    if not candidate.exists():
        print(f"{BANNER} ERROR: no se encontró src/utils.py (ejecuta este script dentro de tu carpeta del proyecto)")
        sys.exit(2)
    patch_utils(str(candidate))

if __name__ == '__main__':
    main()
