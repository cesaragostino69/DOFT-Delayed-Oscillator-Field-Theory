# src/doft/models/model.py
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import math

from doft.utils.utils import spectral_entropy

class DOFTModel:
    def __init__(self, grid_size, a, tau, a_ref, tau_ref, gamma, seed):
        self.grid_size = grid_size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.dtype = np.float64

        # --- Nondimensionalization & Stable Time Step ---
        self.tau_ref = tau_ref
        self.dt_nondim = 0.02  # IMEX is stable with a larger dt
        self.dt = self.dt_nondim * self.tau_ref
        self.a_nondim = a / a_ref
        self.gamma_nondim = gamma * self.tau_ref
        
        # --- Prony Memory Model (Replaces Q_history) ---
        # This implementation follows DOFT v1.2 spec with M=3 auxiliary ODEs
        # It eliminates the large history buffer, solving the memory problem.
        self.prony_M = 3
        # Weights (w_m) and timescales (theta_m) for the memory kernel
        self.prony_weights = np.array([0.5, 0.3, 0.2], dtype=self.dtype)
        base_theta = tau / 3.0 # Tie theta to tau
        self.prony_thetas = np.array([base_theta, base_theta * 10, base_theta * 100], dtype=self.dtype)
        
        # Nondimensionalize Prony parameters
        self.prony_weights_nondim = self.prony_weights * a_ref
        self.prony_thetas_nondim = self.prony_thetas / self.tau_ref

        # State variables
        self.Q = np.zeros((grid_size, grid_size), dtype=self.dtype)
        self.P = np.zeros((grid_size, grid_size), dtype=self.dtype)
        # Auxiliary variables for Prony memory, replacing the giant Q_history
        self.Y = np.zeros((self.prony_M, grid_size, grid_size), dtype=self.dtype)


    def _step_imex(self):
        """
        Advances the simulation using a stable Semi-Implicit (IMEX) Euler method.
        This integrator is robust against stiffness from the damping term.
        """
        # --- 1. Calculate derivatives for auxiliary variables ---
        laplacian_Q = (
            np.roll(self.Q, 1, axis=0) + np.roll(self.Q, -1, axis=0) +
            np.roll(self.Q, 1, axis=1) + np.roll(self.Q, -1, axis=1) - 4 * self.Q
        )
        
        # dY/dt = -Y/theta + ∇²Q
        dY_dt = -self.Y / self.prony_thetas_nondim[:, None, None] + laplacian_Q

        # --- 2. Calculate the total memory force ---
        # Memory Force = a * Σ(w_m * Y_m)
        memory_force = self.a_nondim * np.sum(self.prony_weights_nondim[:, None, None] * self.Y, axis=0)
        
        # --- 3. Update state variables using IMEX scheme ---
        dt = self.dt_nondim

        # Update Y (Prony memory) explicitly
        Y_new = self.Y + dt * dY_dt

        # Update Q explicitly using the *old* P
        Q_new = self.Q + dt * self.P

        # Update P semi-implicitly: the P term is treated implicitly for stability
        # P_new = (P_old + dt*(-Q_new + F_mem)) / (1 + dt*gamma)
        numerator = self.P + dt * (-Q_new + memory_force)
        denominator = 1.0 + dt * self.gamma_nondim
        P_new = numerator / denominator
        
        # --- 4. Assign new states ---
        self.Q, self.P, self.Y = Q_new, P_new, Y_new

    def _calculate_pulse_metrics(self, n_steps):
        # This function remains largely the same, but calls the new integrator
        self.Q.fill(0.0); self.P.fill(0.0); self.Y.fill(0.0)
        center = self.grid_size // 2
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.Q = 0.1 * np.exp(-((x - center)**2 + (y - center)**2) / 10.0)
        
        num_angles = 16
        thetas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        front_detections = {theta: [] for theta in thetas}
        high_thresh, low_thresh = 0.01, 0.007
        is_triggered = {theta: False for theta in thetas}
        max_r_so_far = {theta: 0 for theta in thetas}

        for t_idx in range(n_steps):
            self._step_imex() # <-- Call the new stable integrator
            for theta in thetas:
                for r in range(max_r_so_far[theta], center):
                    px = int(center + r * np.cos(theta)); py = int(center + r * np.sin(theta))
                    val = self.Q[py, px]
                    if not is_triggered[theta] and val > high_thresh: is_triggered[theta] = True
                    elif is_triggered[theta] and val < low_thresh: is_triggered[theta] = False
                    if is_triggered[theta] and r > max_r_so_far[theta]: max_r_so_far[theta] = r
                if max_r_so_far[theta] > 0:
                    front_detections[theta].append((t_idx * self.dt, max_r_so_far[theta]))
        
        c_thetas = []; c_thetas_ci_low, c_thetas_ci_high = [], []
        for theta in thetas:
            detections = np.array(list(dict.fromkeys(front_detections[theta])))
            if len(detections) < 10: continue
            times, dists = detections[:, 0], detections[:, 1]
            res = theilslopes(dists, times, 0.95)
            c_thetas.append(res[0]); c_thetas_ci_low.append(res[2]); c_thetas_ci_high.append(res[3])
            
        if not c_thetas: return {'ceff_pulse': 0.0, 'ceff_pulse_ic95': 0.0, 'anisotropy_max_pct': 100.0,
                                 'var_c_over_c2': 1.0, 'ceff_iso_x': 0.0, 'ceff_iso_y': 0.0, 'ceff_iso_z': 0.0}

        c_thetas = np.array(c_thetas)
        mean_c = np.mean(c_thetas)
        var_c_over_c2 = np.var(c_thetas) / (mean_c**2) if mean_c > 0 else 1.0
        anisotropy_max_pct = (np.max(np.abs(c_thetas - mean_c)) / mean_c) * 100 if mean_c > 0 else 100.0
        ci_width = np.mean(c_thetas_ci_high) - np.mean(c_thetas_ci_low)
        c_x = c_thetas[0]; c_y = c_thetas[num_angles // 4]; c_z = c_thetas[num_angles // 8]
        return {'ceff_pulse': mean_c, 'ceff_pulse_ic95': ci_width / 2.0, 'anisotropy_max_pct': anisotropy_max_pct,
                'var_c_over_c2': var_c_over_c2, 'ceff_iso_x': c_x, 'ceff_iso_y': c_y, 'ceff_iso_z': c_z}

    def _calculate_lpc_metrics(self, n_steps):
        self.Q = self.rng.normal(0, 0.1, self.Q.shape); self.P.fill(0.0); self.Y.fill(0.0)
        center = self.grid_size // 2
        time_series = np.zeros(n_steps)
        for t_idx in range(n_steps):
            self._step_imex() # <-- Call the new stable integrator
            time_series[t_idx] = self.Q[center, center]

        if not np.all(np.isfinite(time_series)):
            print("  WARNING: Non-finite values detected in LPC time series. Skipping metrics.")
            return {'lpc_deltaK_neg_frac': np.nan, 'lpc_brake_count': 0, 'lpc_windows_analyzed': 0}, pd.DataFrame()

        win_size, overlap = 4096, 2048
        step = win_size - overlap
        if len(time_series) < win_size: return {}, pd.DataFrame()
        block_data, last_K = [], None
        num_windows = (len(time_series) - win_size) // step + 1
        for i in range(num_windows):
            window_data = time_series[i*step : i*step + win_size]
            K_metric = spectral_entropy(window_data)
            deltaK = K_metric - last_K if last_K is not None else 0.0
            block_data.append({'window_id': i, 'K_metric': K_metric, 'deltaK': deltaK})
            last_K = K_metric
        if not block_data: return {}, pd.DataFrame()
        blocks_df = pd.DataFrame(block_data)
        
        windows_analyzed = len(blocks_df)
        if windows_analyzed > 1:
            deltaK_neg_count = (blocks_df['deltaK'][1:] <= 0).sum()
            lpc_deltaK_neg_frac = deltaK_neg_count / (windows_analyzed - 1)
        else: lpc_deltaK_neg_frac = 0.0
        return {'lpc_deltaK_neg_frac': lpc_deltaK_neg_frac, 'lpc_brake_count': 0,
                'lpc_windows_analyzed': windows_analyzed}, blocks_df

    def run(self):
        pulse_duration = 150 # Physical time
        lpc_duration = 1500 # Physical time
        pulse_steps = int(pulse_duration / self.dt)
        lpc_steps = int(lpc_duration / self.dt)

        pulse_metrics = self._calculate_pulse_metrics(n_steps=pulse_steps)
        lpc_metrics, blocks_df = self._calculate_lpc_metrics(n_steps=lpc_steps)
        final_run_metrics = {**pulse_metrics, **lpc_metrics}
        return final_run_metrics, blocks_df