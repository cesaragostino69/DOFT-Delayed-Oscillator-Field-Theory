#!/usr/bin/env python3
import re, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def backup(path: Path):
    bk = path.with_suffix(path.suffix + ".bak")
    if not bk.exists():
        shutil.copy2(path, bk)

def patch_requirements():
    path = ROOT / "requirements.txt"
    if not path.exists(): 
        print("[warn] requirements.txt no encontrado, omito")
        return
    txt = path.read_text(encoding="utf-8")
    if "typing-extensions" not in txt:
        print("[patch] requirements.txt → añado typing-extensions>=4.8")
        backup(path)
        txt = txt.rstrip() + "\ntyping-extensions>=4.8\n"
        path.write_text(txt, encoding="utf-8")

def ensure_utils_helpers():
    path = ROOT / "src" / "utils.py"
    if not path.exists():
        print("[warn] src/utils.py no existe, omito helpers")
        return
    txt = path.read_text(encoding="utf-8")
    changed = False

    if "def ensure_numpy(" not in txt:
        changed = True
        txt += """

def ensure_numpy(x):
    \"\"\"Devuelve ndarray de NumPy; acepta listas, ndarray o tensores Torch.\"\"\"
    try:
        import numpy as _np
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            return x.detach().cpu().numpy()
        return _np.asarray(x)
    except Exception:
        import numpy as _np
        return _np.asarray(x)

def _to_numpy(x):
    return ensure_numpy(x)

"""
    if "def spectral_entropy(" not in txt:
        changed = True
        txt += r'''
def spectral_entropy(x):
    """
    Entropía espectral robusta: evita fallas con series cortas o constantes.
    Devuelve float.
    """
    import numpy as np
    x = ensure_numpy(x).reshape(-1)
    n = x.shape[0]
    if n < 4:
        return 0.0
    x = x - np.mean(x)
    if not np.any(np.isfinite(x)):
        return 0.0
    X = np.fft.rfft(x)
    P = (np.abs(X) ** 2)
    s = P.sum()
    if s <= 0 or not np.isfinite(s):
        return 0.0
    p = P / s
    # Evitar log(0)
    p = np.where(p > 0, p, 1e-300)
    H = -np.sum(p * np.log(p))
    # Normalizado por log(len(p))
    Hn = H / max(np.log(len(p)), 1.0)
    if not np.isfinite(Hn):
        return 0.0
    return float(Hn)
'''
    if changed:
        print("[patch] src/utils.py → añadidos helpers ensure_numpy/_to_numpy/spectral_entropy")
        backup(path)
        path.write_text(txt, encoding="utf-8")
    else:
        print("[ok] src/utils.py ya tenía helpers; sin cambios")

def patch_model_sum():
    """
    Reemplaza 'np.sum(self.Y, axis=1)' por suma compatible Torch/NumPy.
    """
    path = ROOT / "src" / "model.py"
    if not path.exists():
        print("[warn] src/model.py no existe, omito")
        return
    txt = path.read_text(encoding="utf-8")
    pattern = r"np\.sum\(\s*self\.Y\s*,\s*axis\s*=\s*1\s*\)"
    repl = "(self.Y.sum(dim=1) if hasattr(self.Y, 'sum') and 'torch' in type(self.Y).__module__ else np.sum(self.Y, axis=1))"
    if re.search(pattern, txt):
        print("[patch] src/model.py → corrigiendo suma sobre self.Y (Torch/NumPy)")
        backup(path)
        txt = re.sub(pattern, repl, txt)
        path.write_text(txt, encoding="utf-8")
    else:
        print("[info] No encontré 'np.sum(self.Y, axis=1)'; nada que reemplazar (quizás ya estaba parcheado).")

def main():
    patch_requirements()
    ensure_utils_helpers()
    patch_model_sum()
    print("[done] Patch v12c aplicado. Recordá: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
