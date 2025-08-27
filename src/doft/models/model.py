# src/doft/models/model.py
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import math

# Import the existing spectral_entropy function from utils
from doft.utils.utils import spectral_entropy

class DOFTModel:
    def __init__(self, grid_size, a, tau, gamma, seed):
        self.grid_size = grid_size
        self.a = a
        self.tau = tau
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
        # Define dt here to make it available for buffer sizing
        self.dt = 0.1

        # AUDIT #0006 BLOCKER FIX: The history buffer size MUST depend on dt.
        # The previous implementation had a fixed small size, causing a critical physics bug.
        self.history_steps = int(math.ceil(self.tau / self.dt)) + 2  # Correct sizing
        
        self.Q = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.P = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.Q_history = np.zeros((self.history_steps, grid_size, grid_size), dtype=np.float64)

    def _step_euler(self, t_idx):
        # Use the class instance's dt
        delay_steps = int(round(self.tau / self.dt))
        history_idx = (t_idx - delay_steps) % self.history_steps
        
        Q_delayed = self.Q_history[history_idx]
        K_term = self.a * (
            np.roll(Q_delayed, 1, axis=0) + np.roll(Q_delayed, -1, axis=0) +
            np.roll(Q_delayed, 1, axis=1) + np.roll(Q_delayed, -1, axis=1) - 4 * Q_delayed
        )
        P_new = self.P + self.dt * (-self.gamma * self.P - self.Q + K_term)
        Q_new = self.Q + self.dt * self.P
        self.P, self.Q = P_new, Q_new
        self.Q_history[t_idx % self.history_steps] = self.Q

    def _calculate_pulse_metrics(self, n_steps):
        """
        Measures c_eff along multiple radial angles to compute all isotropy KPIs.
        """
        # 1. Initialization
        self.Q.fill(0.0); self.P.fill(0.0); self.Q_history.fill(0.0)
        center = self.grid_size // 2
        
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.Q = np.exp(-((x - center)**2 + (y - center)**2) / 10.0)
        
        # 2. Setup for Radial Front Detection
        num_angles = 16
        thetas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        front_detections = {theta: [] for theta in thetas}
        
        high_thresh, low_thresh = 0.1, 0.07
        is_triggered = {theta: False for theta in thetas}
        max_r_so_far = {theta: 0 for theta in thetas}

        # 3. Simulation and Detection
        for t_idx in range(n_steps):
            self._step_euler(t_idx)
            
            for theta in thetas:
                for r in range(max_r_so_far[theta], center):
                    px = int(center + r * np.cos(theta)); py = int(center + r * np.sin(theta))
                    val = self.Q[py, px]
                    
                    if not is_triggered[theta] and val > high_thresh: is_triggered[theta] = True
                    elif is_triggered[theta] and val < low_thresh: is_triggered[theta] = False
                    
                    if is_triggered[theta] and r > max_r_so_far[theta]: max_r_so_far[theta] = r
                
                if max_r_so_far[theta] > 0:
                    front_detections[theta].append((t_idx * self.dt, max_r_so_far[theta]))
        
        # 4. Calculate c_eff per angle and aggregate metrics
        c_thetas = []; c_thetas_ci_low, c_thetas_ci_high = [], []
        for theta in thetas:
            detections = np.array(list(dict.fromkeys(front_detections[theta])))
            if len(detections) < 5: continue
            
            times, dists = detections[:, 0], detections[:, 1]
            res = theilslopes(dists, times, 0.95)
            c_thetas.append(res[0]); c_thetas_ci_low.append(res[2]); c_thetas_ci_high.append(res[3])
            
        if not c_thetas:
            return {'ceff_pulse': 0.0, 'ceff_pulse_ic95': 0.0, 'anisotropy_max_pct': 100.0,
                    'var_c_over_c2': 1.0, 'ceff_iso_x': 0.0, 'ceff_iso_y': 0.0, 'ceff_iso_z': 0.0}

        c_thetas = np.array(c_thetas)
        mean_c = np.mean(c_thetas)
        
        var_c_over_c2 = np.var(c_thetas) / (mean_c**2) if mean_c > 0 else 1.0
        anisotropy_max_pct = (np.max(np.abs(c_thetas - mean_c)) / mean_c) * 100 if mean_c > 0 else 100.0
        ci_width = np.mean(c_thetas_ci_high) - np.mean(c_thetas_ci_low)
        
        # AUDIT #0006 FIX: Use correct angle indices for x, y, and diagonal (z) axes.
        c_x = c_thetas[0]  # Angle 0°
        c_y = c_thetas[num_angles // 4]  # Angle 90° (pi/2)
        c_z = c_thetas[num_angles // 8]  # Angle 45° (pi/4) - DIAGONAL
        
        return {'ceff_pulse': mean_c, 'ceff_pulse_ic95': ci_width / 2.0,
                'anisotropy_max_pct': anisotropy_max_pct, 'var_c_over_c2': var_c_over_c2,
                'ceff_iso_x': c_x, 'ceff_iso_y': c_y, 'ceff_iso_z': c_z}

    def _calculate_lpc_metrics(self, n_steps):
        """
        Implements correct windowing and uses existing spectral_entropy util.
        """
        # 1. Initialization and passive simulation
        self.Q = self.rng.normal(0, 1.0, self.Q.shape); self.P.fill(0.0); self.Q_history.fill(0.0)
        
        center = self.grid_size // 2
        time_series = np.zeros(n_steps)
        for t_idx in range(n_steps):
            self._step_euler(t_idx)
            time_series[t_idx] = self.Q[center, center]

        # 2. Manual windowing and entropy calculation
        win_size, overlap = 2048, 1024
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
        
        # 3. Calculate summary metrics for runs.csv
        windows_analyzed = len(blocks_df)
        if windows_analyzed > 1:
            deltaK_neg_count = (blocks_df['deltaK'][1:] <= 0).sum()
            lpc_deltaK_neg_frac = deltaK_neg_count / (windows_analyzed - 1)
        else:
            lpc_deltaK_neg_frac = 0.0

        # AUDIT #0006 FIX: Correct LPC reporting semantics.
        # 'lpc_vcount' should be 'lpc_brake_count' and must be 0 in passive mode.
        return {
            'lpc_deltaK_neg_frac': lpc_deltaK_neg_frac,
            'lpc_brake_count': 0, # Should be 0 by design in passive runs.
            'lpc_windows_analyzed': windows_analyzed
        }, blocks_df

    def run(self):
        pulse_metrics = self._calculate_pulse_metrics(n_steps=3000)
        lpc_metrics, blocks_df = self._calculate_lpc_metrics(n_steps=30000)
        final_run_metrics = {**pulse_metrics, **lpc_metrics}
        return final_run_metrics, blocks_df