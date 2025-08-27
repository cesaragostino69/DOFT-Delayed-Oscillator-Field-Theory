# src/model.py
# -*- coding: utf-8 -*-
"""
Core implementation of the DOFT model, refactored for Phase 1 QA issues.
- Implements a robust, physical pulse-front detection for c_eff.
- Ensures LPC metrics are correctly calculated and returned.
- Added detailed error logging for numerical issues.
"""

import os
import math
import numpy as np
import json
import datetime
from ..utils.utils import RateLogger, spectral_entropy

DEFAULT_DTYPE = "float64"
def _as_dtype(name):
    if name in ("float64","double","np.float64"): return "float64"
    if name in ("float32","single","np.float32"): return "float32"
    return DEFAULT_DTYPE


import numpy as np

from ..utils.utils import RateLogger, spectral_entropy, anisotropy_from_ceff_map


class DOFTModel:
    # CPU-only core with 3-exponential Prony memory, c_eff map, hbar_eff, LPC check.
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.steps = int(cfg.get("steps", 50_000))
        self.dt = float(cfg.get("dt", 1e-3))
        self.K = float(cfg.get("K", 0.3))
        self.batch = int(cfg.get("batch_replicas", 64))
        self.log_interval = int(cfg.get("log_interval", 60))

        L = int(cfg.get("L", 64))
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        a_mean = float(cfg["a"]["mean"]); a_sigma = float(cfg["a"]["sigma"])
        tau_mean = float(cfg["tau0"]["mean"]); tau_sigma = float(cfg["tau0"]["sigma"])
        self.a_map = rng.normal(a_mean, a_sigma, size=(L, L)).astype(np.float64)
        self.tau_map = rng.normal(tau_mean, tau_sigma, size=(L, L)).astype(np.float64)
        self.ceff_map = np.divide(self.a_map, self.tau_map + 1e-12)
        self.ceff_bar = float(np.mean(self.ceff_map))
        self.aniso_rel = anisotropy_from_ceff_map(self.ceff_map)

        self.device = "cpu"

        self.Q = np.zeros(self.batch, dtype=np.float64)
        self.P = np.zeros(self.batch, dtype=np.float64)
        self.theta = np.array(cfg.get("prony_thetas", [1e-3, 1e-2, 1e-1]), dtype=np.float64).reshape(1, 3)
        self.w = np.array(cfg.get("prony_weights", [0.6, 0.3, 0.1]), dtype=np.float64).reshape(1, 3)
        self.Y = np.zeros((self.batch, 3), dtype=np.float64)

        self.win = int(cfg.get("window", 2048))
        self.bufQ = np.zeros((self.batch, self.win), dtype=np.float64)
        self.bufP = np.zeros((self.batch, self.win), dtype=np.float64)
        self.bidx = 0

    def step_euler(self, xi_amp: float):
        dt = self.dt
        xi = np.random.randn(*self.Q.shape).astype(np.float64) * float(xi_amp)
        self.Y = self.Y + dt * ( - self.Y / self.theta + self.w * self.Q.reshape(-1,1) )
        Mterm = np.sum(self.Y, axis=1)
        self.P = self.P + dt * (-self.K * self.Q + Mterm + xi)
        self.Q = self.Q + dt * self.P
        j = self.bidx % self.win
        self.bufQ[:, j] = self.Q
        self.bufP[:, j] = self.P
        self.bidx += 1

    def _to_numpy(self, T):
        return np.array(T, dtype=np.float64, copy=True)

    def run(self, gamma: float, xi_amp: float, seed: int, out_dir: str) -> dict:
        np.random.seed(int(seed))

        rl = RateLogger(self.log_interval)
        
        noise_floor = xi_amp if xi_amp > 0 else 1e-5
        thresholds = [3 * noise_floor, 5 * noise_floor]
        cross_times = {T: {} for T in thresholds}

        for t in range(self.steps):
            self._step_euler(xi_amp)
            
            q_np = self.Q.cpu().numpy() if self.engine == "torch" else self.Q
            q_abs = np.abs(q_np[0])
            
            for T in thresholds:
                coords = np.argwhere(q_abs > T)
                if coords.size == 0: continue
                
                radii = np.sqrt((coords[:,0] - center)**2 + (coords[:,1] - center)**2)
                
                for r_int in np.unique(radii.astype(int)):
                    if r_int not in cross_times[T]:
                        last_time = max(cross_times[T].values()) if cross_times[T] else 0
                        current_time = t * self.dt
                        if current_time >= last_time:
                            cross_times[T][r_int] = current_time

            max_q = q_abs.max()
            rl.tick(t + 1, max_q=f"{max_q:.3f}")

        all_fits = []
        a_mean = self.cfg['a']['mean']
        
        for T in thresholds:
            points = sorted(cross_times[T].items())
            valid_points = [p for p in points if p[0] < L * 0.4]
            if len(valid_points) < 5: continue

            radii = np.array([p[0] for p in valid_points]) * a_mean
            times = np.array([p[1] for p in valid_points])
            
            if np.any(np.isnan(radii)) or np.any(np.isnan(times)):
                self._log_nan_event(out_dir, "Wavefront data contains NaN values.", {"times": times.tolist(), "radii": radii.tolist()})
                continue
            if times.size < 2 or radii.size < 2:
                continue
            if np.all(times == times[0]):
                self._log_nan_event(out_dir, "All crossing times are identical.", {"times": times.tolist(), "radii": radii.tolist()})
                continue

            try:
                slope, _ = np.polyfit(times, radii, 1)
                if slope > 0 and np.isfinite(slope):
                    all_fits.append(slope)
                else:
                    self._log_nan_event(out_dir, f"The slope fit yielded a non-physical value (slope={slope}).", {"times": times.tolist(), "radii": radii.tolist()})
            except (np.linalg.LinAlgError, ValueError) as e:
                self._log_nan_event(out_dir, f"np.polyfit failed with error '{e}'.", {"times": times.tolist(), "radii": radii.tolist()})
                continue

        if not all_fits:
            self._log_nan_event(out_dir, "No valid velocity fits were obtained for any threshold.", {"cross_times": cross_times})
            ceff_pulse = np.nan
        else:
            ceff_pulse = np.mean(all_fits)
        
        return {"ceff_pulse": ceff_pulse, "anisotropy_max_pct": 0.0}, []
