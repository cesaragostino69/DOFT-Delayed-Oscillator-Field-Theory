#!/usr/bin/env python3
import re, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def backup(p: Path):
    b = p.with_suffix(p.suffix + ".bak")
    if not b.exists():
        shutil.copy2(p, b)

def patch_requirements():
    req = ROOT / "requirements.txt"
    if not req.exists():
        print("[warn] requirements.txt no encontrado")
        return
    txt = req.read_text(encoding="utf-8")
    need = []
    if "typing-extensions" not in txt:
        need.append("typing-extensions>=4.8")
    if need:
        print("[patch] requirements.txt ->", ", ".join(need))
        backup(req)
        req.write_text(txt.rstrip() + "\n" + "\n".join(need) + "\n", encoding="utf-8")

def ensure_utils_helpers():
    u = ROOT / "src" / "utils.py"
    if not u.exists():
        print("[warn] src/utils.py no existe; omito helpers")
        return
    txt = u.read_text(encoding="utf-8")
    changed = False
    if "def ensure_numpy(" not in txt:
        txt += """
def ensure_numpy(x):
    \"\"\"Devuelve ndarray de NumPy; acepta listas, ndarray o tensores Torch.\"\"\"
    import numpy as _np
    try:
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            return x.detach().cpu().numpy()
        return _np.asarray(x, dtype=_np.float64)
    except Exception:
        return _np.asarray(x, dtype=_np.float64)

def _to_numpy(x):
    return ensure_numpy(x)
"""
        changed = True
    if "def spectral_entropy(" not in txt:
        txt += r'''
def spectral_entropy(x):
    """
    Entropía espectral robusta para series cortas/constantes.
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
    p = np.where(p > 0, p, 1e-300)  # evita log(0)
    H = -np.sum(p * np.log(p))
    Hn = H / max(np.log(len(p)), 1.0)
    return float(Hn) if np.isfinite(Hn) else 0.0
'''
        changed = True
    if changed:
        print("[patch] src/utils.py -> helpers añadidos")
        backup(u)
        u.write_text(txt, encoding="utf-8")
    else:
        print("[ok] src/utils.py ya tenía helpers")

def patch_model_sum_and_precision():
    m = ROOT / "src" / "model.py"
    if not m.exists():
        print("[warn] src/model.py no existe; omito")
        return
    txt = m.read_text(encoding="utf-8")
    changed = False

    # (A) Suma Torch/NumPy
    pat = r"np\.sum\(\s*self\.Y\s*,\s*axis\s*=\s*1\s*\)"
    rep = "(self.Y.sum(dim=1) if hasattr(self.Y, 'sum') and 'torch' in type(self.Y).__module__ else np.sum(self.Y, axis=1))"
    if re.search(pat, txt):
        txt = re.sub(pat, rep, txt)
        print("[patch] src/model.py -> suma Torch/NumPy corregida")
        changed = True

    # (B) Fuerza FP64 y preflight log
    if "class DOFTModel" in txt and "def __init__(" in txt:
        if "DEFAULT_DTYPE" not in txt:
            inject = """
DEFAULT_DTYPE = "float64"
def _as_dtype(name):
    if name in ("float64","double","np.float64"): return "float64"
    if name in ("float32","single","np.float32"): return "float32"
    return DEFAULT_DTYPE
"""
            txt = inject + "\n" + txt
            changed = True

        # Inserta configuración de dtype/device y preflight después de leer cfg
        if "self.device =" not in txt:
            # heurística: ponlo cerca de comienzo de __init__
            txt = re.sub(
                r"(class DOFTModel[^\n]*\n\s+def __init__\(self, cfg[^\)]*\):\n)",
                r"""\1        # --- precisión / dispositivo ---
        import numpy as np
        self.engine = cfg.get("engine", "torch")
        self.dtype_name = _as_dtype(str(cfg.get("dtype", "float64")))
        if self.dtype_name == "float64":
            np_dtype = np.float64
        else:
            np_dtype = np.float32

        self.use_gpu = bool(cfg.get("use_gpu", True))
        self.device = "cpu"
        try:
            if self.engine == "torch":
                import torch
                if self.use_gpu and torch.cuda.is_available():
                    self.device = "cuda"
                # fija dtype por defecto en Torch
                torch.set_default_dtype(torch.float64 if self.dtype_name=="float64" else torch.float32)
                # opcional determinismo
                torch.backends.cudnn.deterministic = True
                try: torch.use_deterministic_algorithms(False)
                except Exception: pass
        except Exception:
            pass

        self._np_dtype = np_dtype
""",
                txt
            )
            print("[patch] src/model.py -> bloque dtype/device + determinismo")
            changed = True

        # Asegura que a/tau0 se crean con dtype correcto (NumPy o Torch)
        if "self.a =" in txt and "self.tau0 =" in txt:
            # no garantizamos el patrón exacto; añadimos un post-bloque de cast
            if "def _cast_params_fp(" not in txt:
                post = """

    def _cast_params_fp(self):
        \"\"\"Asegura dtype consistente en a, tau0 y buffers.\"\"\"
        import numpy as np
        if self.engine == "torch":
            import torch
            def to_t(x):
                if hasattr(x, "to"):  # tensor
                    return x.to(device=self.device, dtype=torch.float64 if self.dtype_name=="float64" else torch.float32)
                # array/list
                return torch.as_tensor(np.asarray(x, dtype=np.float64 if self.dtype_name=="float64" else np.float32),
                                       device=self.device)
            if hasattr(self, "a"): self.a = to_t(self.a)
            if hasattr(self, "tau0"): self.tau0 = to_t(self.tau0)
            if hasattr(self, "Y"): self.Y = to_t(self.Y)
        else:
            # NumPy: fuerza dtype
            import numpy as np
            if hasattr(self, "a"): self.a = np.asarray(self.a, dtype=self._np_dtype)
            if hasattr(self, "tau0"): self.tau0 = np.asarray(self.tau0, dtype=self._np_dtype)
            if hasattr(self, "Y"): self.Y = np.asarray(self.Y, dtype=self._np_dtype)
"""
                txt = re.sub(r"(def __init__\([^\n]+\):\n)", r"\1", txt)
                if "_cast_params_fp(self):" not in txt:
                    # Añadir la función antes de métodos siguientes
                    txt = re.sub(r"(\n\s*def [a-zA-Z_]+\()", post + r"\1", txt, count=1)
                changed = True

            if "preflight" not in txt:
                preflight = r"""
        # --- preflight: coherencia de escala ---
        try:
            self._cast_params_fp()
            import numpy as _np
            a_mean = float(_np.mean(self.a.detach().cpu().numpy() if hasattr(self.a, "detach") else self.a))
            t_mean = float(_np.mean(self.tau0.detach().cpu().numpy() if hasattr(self.tau0, "detach") else self.tau0))
            ceff0 = float(_np.mean(
                (self.a.detach().cpu().numpy() if hasattr(self.a,"detach") else self.a) /
                ((_np.asarray(self.tau0.detach().cpu().numpy() if hasattr(self.tau0,"detach") else self.tau0))+1e-12)
            ))
            print(f"[preflight] engine:{self.engine} device:{self.device} dtype:{self.dtype_name}  a_mean:{a_mean:.6g}  tau0_mean:{t_mean:.6g}  c_eff_bar0:{ceff0:.6g}")
            if not (0.5 <= ceff0 <= 5.0):
                print(f"[WARN] c_eff_bar0 fuera de rango esperado [0.5,5] → {ceff0:.6g}. Revisa 'a.mean' y 'tau0.mean' en tu config.")
        except Exception as _e:
            print(f"[preflight] aviso: no pude calcular chequeo inicial ({_e})")
"""
                txt = re.sub(r"(self\._np_dtype = [^\n]+\n)", r"\1" + preflight, txt)
                changed = True

    if changed:
        backup(m)
        m.write_text(txt, encoding="utf-8")
    else:
        print("[info] src/model.py sin cambios (ya parcheado)")

def main():
    patch_requirements()
    ensure_utils_helpers()
    patch_model_sum_and_precision()
    print("[done] Patch v12d aplicado. Volvé a instalar deps:  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
