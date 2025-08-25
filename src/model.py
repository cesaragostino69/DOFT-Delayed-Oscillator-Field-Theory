
DEFAULT_DTYPE = "float64"
def _as_dtype(name):
    if name in ("float64","double","np.float64"): return "float64"
    if name in ("float32","single","np.float32"): return "float32"
    return DEFAULT_DTYPE


import os, math, numpy as np
try:
    import torch
except Exception:
    torch = None

from .utils import RateLogger, spectral_entropy, anisotropy_from_ceff_map


class DOFTModel:
    # GPU (PyTorch) core with 3-exponential Prony memory, c_eff map, hbar_eff, LPC check.
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.steps = int(cfg.get("steps", 120_000))
        self.dt = float(cfg.get("dt", 1e-3))
        self.K = float(cfg.get("K", 0.3))
        self.batch = int(cfg.get("batch_replicas", 64))
        self.log_interval = int(cfg.get("log_interval", 60))

        L = int(cfg.get("L", 64))
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        a_mean = float(cfg["a"]["mean"]); a_sigma = float(cfg["a"]["sigma"])
        tau_mean = float(cfg["tau0"]["mean"]); tau_sigma = float(cfg["tau0"]["sigma"])
        self.a_map = rng.normal(a_mean, a_sigma, size=(L, L)).astype(np.float32)
        self.tau_map = rng.normal(tau_mean, tau_sigma, size=(L, L)).astype(np.float32)
        self.ceff_map = np.divide(self.a_map, self.tau_map + 1e-12)
        self.ceff_bar = float(np.mean(self.ceff_map))
        self.aniso_rel = anisotropy_from_ceff_map(self.ceff_map)

        self.use_gpu = bool(int(os.environ.get("USE_GPU", "1")))
        self.device = "cuda" if (self.use_gpu and torch is not None and torch.cuda.is_available()) else "cpu"
        if torch is None and self.use_gpu:
            raise RuntimeError("GPU requested but torch is not available on this machine.")

        self.dtype = torch.float32 if torch is not None else None

        if torch is not None:
            self.Q = torch.zeros(self.batch, device=self.device, dtype=self.dtype)
            self.P = torch.zeros(self.batch, device=self.device, dtype=self.dtype)
            th = cfg.get("prony_thetas", [1e-3, 1e-2, 1e-1])
            wm = cfg.get("prony_weights", [0.6, 0.3, 0.1])
            assert len(th) == 3 and len(wm) == 3, "Use M=3 for Prony."
            self.theta = torch.tensor(th, device=self.device, dtype=self.dtype).view(1, 3)
            self.w = torch.tensor(wm, device=self.device, dtype=self.dtype).view(1, 3)
            self.Y = torch.zeros(self.batch, 3, device=self.device, dtype=self.dtype)
        else:
            self.Q = np.zeros(self.batch, dtype=np.float32)
            self.P = np.zeros(self.batch, dtype=np.float32)
            self.theta = np.array(cfg.get("prony_thetas", [1e-3, 1e-2, 1e-1]), dtype=np.float32).reshape(1, 3)
            self.w = np.array(cfg.get("prony_weights", [0.6, 0.3, 0.1]), dtype=np.float32).reshape(1, 3)
            self.Y = np.zeros((self.batch, 3), dtype=np.float32)

        self.win = int(cfg.get("window", 2048))
        if self.device == "cuda":
            self.bufQ = torch.zeros(self.batch, self.win, device=self.device, dtype=self.dtype)
            self.bufP = torch.zeros(self.batch, self.win, device=self.device, dtype=self.dtype)
        else:
            self.bufQ = np.zeros((self.batch, self.win), dtype=np.float32)
            self.bufP = np.zeros((self.batch, self.win), dtype=np.float32)
        self.bidx = 0

    def step_euler(self, xi_amp: float):
        dt = self.dt
        if self.device == "cuda":
            xi = torch.randn_like(self.Q) * float(xi_amp)
            self.Y = self.Y + dt * ( - self.Y / self.theta + self.w * self.Q.view(-1,1) )
            Mterm = torch.sum(self.Y, dim=1)
            self.P = self.P + dt * (-self.K * self.Q + Mterm + xi)
            self.Q = self.Q + dt * self.P
            j = self.bidx % self.win
            self.bufQ[:, j] = self.Q
            self.bufP[:, j] = self.P
            self.bidx += 1
        else:
            xi = np.random.randn(*self.Q.shape).astype(np.float32) * float(xi_amp)
            self.Y = self.Y + dt * ( - self.Y / self.theta + self.w * self.Q.reshape(-1,1) )
            Mterm = (torch.sum(self.Y, dim=1) if isinstance(self.Y, torch.Tensor) else (self.Y.sum(dim=1) if hasattr(self.Y, 'sum') and 'torch' in type(self.Y).__module__ else (self.Y.sum(dim=1) if hasattr(self.Y, 'sum') and 'torch' in type(self.Y).__module__ else np.sum(self.Y, axis=1))))
            self.P = self.P + dt * (-self.K * self.Q + Mterm + xi)
            self.Q = self.Q + dt * self.P
            j = self.bidx % self.win
            self.bufQ[:, j] = self.Q
            self.bufP[:, j] = self.P
            self.bidx += 1

    def _to_numpy(self, T):
        if isinstance(T, np.ndarray):
            return T.copy()
        else:
            return T.detach().to("cpu").numpy()

    def run(self, gamma: float, xi_amp: float, seed: int, out_dir: str) -> dict:
        if self.device == "cuda":
            torch.manual_seed(int(seed))
        else:
            np.random.seed(int(seed))

        rl = RateLogger(self.log_interval)
        samples = 0
        lpc_viol_count = 0
        H_start = None

        for t in range(self.steps):
            self.step_euler(xi_amp=xi_amp)
            if self.bidx >= self.win and H_start is None:
                Qnp = self._to_numpy(self.bufQ)
                H_list = [spectral_entropy(Qnp[i, -self.win:]) for i in range(min(16, self.batch))]
                H_start = float(np.nanmean(H_list))
            if (t + 1) % 1024 == 0:
                samples += 1
                Qnp = self._to_numpy(self.bufQ)
                H_list = [spectral_entropy(Qnp[i, -self.win:]) for i in range(min(16, self.batch))]
                H_now = float(np.nanmean(H_list))
                if H_start is not None and (H_now > H_start + 1e-3):
                    lpc_viol_count += 1
            rl.tick(t+1,
                    xi=xi_amp, gamma=gamma, K=self.K,
                    z=0.0, samples=samples, lpcV=lpc_viol_count, ceff_bar=self.ceff_bar)

        Qnp = self._to_numpy(self.bufQ)
        Pnp = self._to_numpy(self.bufP)
        q_std = float(np.std(Qnp[:, -self.win:].ravel()))
        p_std = float(np.std(Pnp[:, -self.win:].ravel()))
        hbar_eff = q_std * p_std

        row = dict(
            gamma=float(gamma),
            xi=float(xi_amp),
            seed=int(seed),
            ceff_bar=float(self.ceff_bar),
            anisotropy_rel=float(self.aniso_rel),
            hbar_eff=float(hbar_eff),
            lpc_viol_frac=float(lpc_viol_count / max(1, samples)),
            steps=int(self.steps),
            dt=float(self.dt),
            batch=int(self.batch),
            device=self.device,
        )
        return row
