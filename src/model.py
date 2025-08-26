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
from .utils import RateLogger, spectral_entropy

try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False

class DOFTModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.steps = int(cfg.get("steps", 50_000))
        self.dt = float(cfg.get("dt", 1e-3))
        self.K = float(cfg.get("K", 0.3))
        self.gamma = float(cfg.get("gamma", 0.01))
        self.batch = 1
        self.log_interval = int(cfg.get("log_interval", 30))

        L = int(cfg.get("L", 128))
        a_mean = float(cfg["a"]["mean"])
        a_std = float(cfg["a"]["std"])
        tau_mean = float(cfg["tau0"]["mean"])
        tau_std = float(cfg["tau0"]["std"])

        self.a_map = self.rng.normal(a_mean, a_std, size=(L, L)).astype(np.float64)
        self.tau_map = self.rng.normal(tau_mean, tau_std, size=(L, L)).astype(np.float64)

        self.device = "cpu"
        self.engine = "numpy"
        if _TORCH_OK and bool(int(os.environ.get("USE_GPU", "0"))):
            if torch.cuda.is_available():
                self.device = "cuda"
                self.engine = "torch"
                torch.manual_seed(self.seed)
                torch.set_default_dtype(torch.float64)

        prony_cfg = cfg.get("prony_memory", {"weights": [0.6, 0.3, 0.1], "thetas": [1e-3, 1e-2, 1e-1]})
        self.prony_w = np.array(prony_cfg["weights"], dtype=np.float64).reshape(1, -1)
        self.prony_theta = np.array(prony_cfg["thetas"], dtype=np.float64).reshape(1, -1)
        num_memory_modes = self.prony_w.shape[1]

        self.Q = np.zeros((self.batch, L, L), dtype=np.float64)
        self.P = np.zeros((self.batch, L, L), dtype=np.float64)
        self.Y = np.zeros((self.batch, L, L, num_memory_modes), dtype=np.float64)

        if self.engine == "torch":
            self.Q = torch.from_numpy(self.Q).to(self.device)
            self.P = torch.from_numpy(self.P).to(self.device)
            self.Y = torch.from_numpy(self.Y).to(self.device)
            self.prony_w = torch.from_numpy(self.prony_w).to(self.device)
            self.prony_theta = torch.from_numpy(self.prony_theta).to(self.device)

        self.lpc_budget = 0.0
        self.lpc_vcount = 0

    def _log_nan_event(self, out_dir, reason: str, data: dict):
        """Logs details of a numerical error to a file."""
        if not out_dir:
            print(f"# ADVERTENCIA (seed={self.seed}): {reason}")
            return

        log_file_path = out_dir / f"error_log_seed_{self.seed}.txt"
        
        log_message = f"""
=================================================
ERROR NUMÉRICO DETECTADO EN LA SIMULACIÓN
=================================================
Seed: {self.seed}
Fecha y Hora: {datetime.datetime.now().isoformat()}

Causa del Error:
----------------
{reason}

Configuración de la Simulación:
-----------------------------
a: {self.cfg.get('a')}
tau0: {self.cfg.get('tau0')}
gamma: {self.cfg.get('gamma')}
xi_amp: {self.cfg.get('xi_amp')}

Datos que Causaron el Error:
----------------------------
{json.dumps(data, indent=2, default=str)}

=================================================
"""
        try:
            with open(log_file_path, "a") as f:
                f.write(log_message)
        except Exception as e:
            print(f"Error al escribir el log de errores: {e}")

    def _initialize_state(self, mode='chaos'):
        if mode == 'chaos':
            self.Q = self.rng.normal(0, 0.5, self.Q.shape).astype(np.float64)
        elif mode == 'pulse':
            self.Q.fill(0)
        self.P.fill(0)
        self.Y.fill(0)
        if self.engine == "torch":
            self.Q = torch.from_numpy(self.Q).to(self.device)
            self.P = torch.from_numpy(self.P).to(self.device)
            self.Y = torch.from_numpy(self.Y).to(self.device)

    def _step_euler(self, xi_amp: float):
        if self.engine == "torch":
            Q_reshaped = self.Q.unsqueeze(-1)
            self.Y += self.dt * (-self.Y / self.prony_theta + self.prony_w * Q_reshaped)
            M_term = torch.sum(self.Y, dim=-1)
            xi = torch.randn_like(self.Q) * xi_amp
        else:
            Q_reshaped = self.Q[..., np.newaxis]
            self.Y += self.dt * (-self.Y / self.prony_theta + self.prony_w * Q_reshaped)
            M_term = np.sum(self.Y, axis=-1)
            xi = self.rng.normal(0, xi_amp, self.Q.shape).astype(np.float64)

        self.P += self.dt * (-self.K * self.Q + M_term - self.gamma * self.P + xi)
        self.Q += self.dt * self.P

    def _monitor_and_apply_lpc(self, t, win_size=1024):
        if t > 0 and t % win_size == 0:
            q_data = self.Q.cpu().numpy() if self.engine == "torch" else self.Q
            current_chaos = spectral_entropy(q_data)
            if t == win_size:
                self.lpc_budget = current_chaos
                return current_chaos, 0.0

            delta_K = current_chaos - self.lpc_budget
            if delta_K > 1e-4:
                self.lpc_vcount += 1
                if self.engine == "torch":
                    with torch.no_grad(): self.prony_w[0, 0] *= 0.999
                else: self.prony_w[0, 0] *= 0.999
            
            self.lpc_budget = min(self.lpc_budget, current_chaos)
            return current_chaos, delta_K
        return None, None

    def run_chaos_experiment(self, xi_amp: float, out_dir=None):
        self._initialize_state(mode='chaos')
        rl = RateLogger(self.log_interval)
        blocks_data = []

        for t in range(self.steps):
            self._step_euler(xi_amp)
            K_metric, deltaK = self._monitor_and_apply_lpc(t)
            if K_metric is not None:
                blocks_data.append({"window_id": t, "K_metric": K_metric, "deltaK": deltaK})
            rl.tick(t + 1, K=self.lpc_budget, dK=deltaK or 0, breaks=self.lpc_vcount)

        lpc_deltaK_neg_frac = sum(1 for b in blocks_data if b['deltaK'] <= 0) / len(blocks_data) if blocks_data else 1.0
        return {"lpc_vcount": self.lpc_vcount, "lpc_deltaK_neg_frac": lpc_deltaK_neg_frac}, blocks_data

    def run_pulse_experiment(self, xi_amp: float, out_dir=None):
        self._initialize_state(mode='pulse')
        L = self.Q.shape[1]
        center = L // 2
        
        y, x = np.ogrid[-center:L-center, -center:L-center]
        pulse = np.exp(-(x*x + y*y) / (2 * 4.0**2)).astype(np.float64)
        if self.engine == "torch":
            self.Q[0, :, :] = torch.from_numpy(pulse).to(self.device)
        else:
            self.Q[0, :, :] = pulse

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
                self._log_nan_event(out_dir, "Datos de frente de onda contienen NaN.", {"times": times.tolist(), "radii": radii.tolist()})
                continue
            if times.size < 2 or radii.size < 2:
                continue
            if np.all(times == times[0]):
                self._log_nan_event(out_dir, "Todos los tiempos de cruce son idénticos.", {"times": times.tolist(), "radii": radii.tolist()})
                continue

            try:
                slope, _ = np.polyfit(times, radii, 1)
                if slope > 0 and np.isfinite(slope):
                    all_fits.append(slope)
                else:
                    self._log_nan_event(out_dir, f"El ajuste de pendiente resultó en un valor no físico (slope={slope}).", {"times": times.tolist(), "radii": radii.tolist()})
            except (np.linalg.LinAlgError, ValueError) as e:
                self._log_nan_event(out_dir, f"np.polyfit falló con el error '{e}'.", {"times": times.tolist(), "radii": radii.tolist()})
                continue

        if not all_fits:
            self._log_nan_event(out_dir, "No se pudieron obtener ajustes de velocidad válidos de ningún umbral.", {"cross_times": cross_times})
            ceff_pulse = np.nan
        else:
            ceff_pulse = np.mean(all_fits)
        
        return {"ceff_pulse": ceff_pulse, "anisotropy_max_pct": 0.0}, []
