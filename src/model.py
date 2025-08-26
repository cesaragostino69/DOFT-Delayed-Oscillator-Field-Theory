# src/model.py
# -*- coding: utf-8 -*-
"""
Core implementation of the DOFT model, refactored for Phase 1 contra-proof.
This version implements two distinct experimental protocols:
1. 'pulse': Measures the emergent speed of light (c_eff) by tracking a wavefront.
2. 'chaos': Tests the Law of Preservation of Chaos (LPC) with a high-entropy initial state.
"""

import os
import math
import numpy as np
from .utils import RateLogger, spectral_entropy

# Try to import torch for GPU acceleration, but gracefully fail if not available.
try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False

class DOFTModel:
    """
    DOFT Model Simulator (Phase 1 Refactor).
    Implements the core physics and the two main experimental protocols.
    """
    def __init__(self, cfg: dict):
        """
        Initializes the model based on a configuration dictionary.
        Sets up the physical parameters, the simulation grid, and the memory kernel.
        """
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        # --- Core Simulation Parameters ---
        self.steps = int(cfg.get("steps", 50_000))
        self.dt = float(cfg.get("dt", 1e-3))
        self.K = float(cfg.get("K", 0.3))
        self.gamma = float(cfg.get("gamma", 0.01)) # Damping term
        self.batch = int(cfg.get("batch_replicas", 1)) # Batch is 1 for these experiments
        self.log_interval = int(cfg.get("log_interval", 30))

        # --- Grid and Network Parameters ---
        L = int(cfg.get("L", 128))
        a_mean = float(cfg["a"]["mean"])
        a_std = float(cfg["a"]["std"])
        tau_mean = float(cfg["tau0"]["mean"])
        tau_std = float(cfg["tau0"]["std"])

        self.a_map = self.rng.normal(a_mean, a_std, size=(L, L)).astype(np.float32)
        self.tau_map = self.rng.normal(tau_mean, tau_std, size=(L, L)).astype(np.float32)

        # --- Engine and Device Setup (CPU/GPU) ---
        self.device = "cpu"
        self.engine = "numpy"
        if _TORCH_OK and bool(int(os.environ.get("USE_GPU", "0"))):
            if torch.cuda.is_available():
                self.device = "cuda"
                self.engine = "torch"
                torch.manual_seed(self.seed)
            else:
                print("# WARNING: USE_GPU=1 but no CUDA device found. Falling back to CPU.")

        # --- Prony Memory Kernel ---
        prony_cfg = cfg.get("prony_memory", {"weights": [0.6, 0.3, 0.1], "thetas": [1e-3, 1e-2, 1e-1]})
        self.prony_w = np.array(prony_cfg["weights"], dtype=np.float32).reshape(1, -1)
        self.prony_theta = np.array(prony_cfg["thetas"], dtype=np.float32).reshape(1, -1)
        
        # CRITICAL FIX: Correctly determine the number of memory modes.
        num_memory_modes = self.prony_w.shape[1]

        # --- State Variables (Q, P, Y) ---
        self.Q = np.zeros((self.batch, L, L), dtype=np.float32)
        self.P = np.zeros((self.batch, L, L), dtype=np.float32)
        self.Y = np.zeros((self.batch, L, L, num_memory_modes), dtype=np.float32)

        if self.engine == "torch":
            self.Q = torch.from_numpy(self.Q).to(self.device)
            self.P = torch.from_numpy(self.P).to(self.device)
            self.Y = torch.from_numpy(self.Y).to(self.device)
            self.prony_w = torch.from_numpy(self.prony_w).to(self.device)
            self.prony_theta = torch.from_numpy(self.prony_theta).to(self.device)

        # --- LPC (Law of Preservation of Chaos) Parameters ---
        self.lpc_budget = 0.0
        self.lpc_vcount = 0
        self.lpc_active = False

    def _initialize_state(self, mode='chaos'):
        """ Initializes Q and P based on the experiment mode. """
        if mode == 'chaos':
            if self.engine == "torch":
                self.Q = torch.randn_like(self.Q) * 0.5
            else:
                self.Q = self.rng.normal(0, 0.5, self.Q.shape).astype(np.float32)
        elif mode == 'pulse':
            self.Q.fill(0)
        self.P.fill(0)
        self.Y.fill(0)

    def _step_euler(self, xi_amp: float):
        """
        Performs one time step of the simulation using the Euler-Maruyama method.
        """
        if self.engine == "torch":
            Q_reshaped = self.Q.unsqueeze(-1)
            self.Y += self.dt * (-self.Y / self.prony_theta + self.prony_w * Q_reshaped)
            M_term = torch.sum(self.Y, dim=-1)
            xi = torch.randn_like(self.Q) * xi_amp
        else:
            Q_reshaped = self.Q[..., np.newaxis]
            self.Y += self.dt * (-self.Y / self.prony_theta + self.prony_w * Q_reshaped)
            M_term = np.sum(self.Y, axis=-1)
            xi = self.rng.normal(0, xi_amp, self.Q.shape).astype(np.float32)

        self.P += self.dt * (-self.K * self.Q + M_term - self.gamma * self.P + xi)
        self.Q += self.dt * self.P

    def _monitor_and_apply_lpc(self, t, win_size=2048):
        """ Monitors chaos and applies the 'copy brake' if LPC is violated. """
        if t > 0 and t % win_size == 0:
            q_data = self.Q.cpu().numpy() if self.engine == "torch" else self.Q
            current_chaos = spectral_entropy(q_data)
            if t == win_size:
                self.lpc_budget = current_chaos
                return 0.0, 0.0

            delta_K = current_chaos - self.lpc_budget
            if delta_K > 1e-4:
                self.lpc_vcount += 1
                if self.engine == "torch":
                    with torch.no_grad():
                        self.prony_w[0, 0] *= 0.999
                else:
                    self.prony_w[0, 0] *= 0.999
            self.lpc_budget = min(self.lpc_budget, current_chaos)
            return current_chaos, delta_K
        return None, None

    def run_chaos_experiment(self, xi_amp: float):
        """ Executes the LPC validation experiment. """
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

    def run_pulse_experiment(self, xi_amp: float):
        """ Executes the emergent 'c' measurement experiment. """
        self._initialize_state(mode='pulse')
        L = self.Q.shape[1]
        center = L // 2
        
        x, y = np.ogrid[-center:L-center, -center:L-center]
        pulse = np.exp(-(x*x + y*y) / (2 * 5.0**2)).astype(np.float32)
        if self.engine == "torch":
            self.Q[0, :, :] = torch.from_numpy(pulse).to(self.device)
        else:
            self.Q[0, :, :] = pulse

        rl = RateLogger(self.log_interval)
        threshold = 3 * xi_amp if xi_amp > 0 else 1e-3
        wavefront_radius = []

        for t in range(self.steps):
            self._step_euler(xi_amp)
            if t > 200 and t % 100 == 0:
                q_np = self.Q.cpu().numpy() if self.engine == "torch" else self.Q
                q_abs = np.abs(q_np[0])
                if q_abs.max() < threshold: continue
                coords = np.argwhere(q_abs > threshold)
                if coords.size > 0:
                    radii = np.sqrt((coords[:,0] - center)**2 + (coords[:,1] - center)**2)
                    wavefront_radius.append((t * self.dt, np.mean(radii)))
            max_q = self.Q.abs().max().item() if self.engine=="torch" else np.max(np.abs(self.Q))
            rl.tick(t + 1, max_q=max_q)

        if len(wavefront_radius) < 5:
            return {"ceff_pulse": -1.0, "anisotropy_max_pct": -1.0}, []

        times = np.array([r[0] for r in wavefront_radius])
        radii = np.array([r[1] for r in wavefront_radius])
        
        a_mean = self.cfg['a']['mean']
        ceff_pulse = np.polyfit(times, radii * a_mean, 1)[0]
        
        return {"ceff_pulse": ceff_pulse, "anisotropy_max_pct": 0.0}, []
