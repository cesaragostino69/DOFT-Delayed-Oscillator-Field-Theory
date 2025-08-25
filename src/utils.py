
import time, math, numpy as np

class RateLogger:
    def __init__(self, interval_sec=60):
        self.t0 = time.time()
        self.last = self.t0
        self.interval = interval_sec
        self.step = 0

    def tick(self, step, **kv):
        self.step = step
        now = time.time()
        if now - self.last >= self.interval:
            dt = int(now - self.t0)
            rate = (step / max(1, dt))
            labels = []
            for k, v in kv.items():
                if isinstance(v, float):
                    if abs(v) >= 1e3 or (abs(v) > 0 and abs(v) < 1e-2):
                        labels.append(f"{k}:{v:.3e}")
                    else:
                        labels.append(f"{k}:{v:.3f}")
                else:
                    labels.append(f"{k}:{v}")
            lbl = ", ".join(labels)
            print(f"t:{dt:6d}s, step:{step}, rate:{rate:.1f}/s, {lbl}", flush=True)


def spectral_entropy(x: np.ndarray) -> float:
    # Simple spectral entropy on a 1D real signal (robust to short input).
    if x is None or len(x) < 8:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if np.allclose(x, 0):
        return 0.0
    X = np.fft.rfft(x)
    P = (np.abs(X) ** 2)
    s = P.sum()
    if s <= 0:
        return 0.0
    p = P / s
    p = np.clip(p, 1e-16, 1.0)
    H = -np.sum(p * np.log(p))
    return float(H)


def anisotropy_from_ceff_map(ceff_map: np.ndarray) -> float:
    # Delta c over c using directional means along x and y.
    if ceff_map.ndim != 2:
        return 0.0
    cx = np.mean(ceff_map, axis=0).mean()
    cy = np.mean(ceff_map, axis=1).mean()
    cbar = 0.5 * (cx + cy)
    if cbar == 0:
        return 0.0
    return float(abs(cx - cy) / cbar)



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



def ensure_numpy(x):
    """Devuelve ndarray de NumPy; acepta listas, ndarray o tensores Torch."""
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

