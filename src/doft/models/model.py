# src/doft/models/model.py
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

# Import the existing, correct spectral_entropy function from utils
from doft.utils.utils import spectral_entropy

class DOFTModel:
    def __init__(self, grid_size, a, tau, gamma, seed):
        self.grid_size = grid_size
        self.a = a
        self.tau = tau
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.Q = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.P = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.history_steps = int(np.ceil(self.tau * 2)) + 1
        self.Q_history = np.zeros((self.history_steps, grid_size, grid_size), dtype=np.float64)

    def _step_euler(self, dt, t_idx):
        history_idx = (t_idx - int(self.tau / dt)) % self.history_steps
        Q_delayed = self.Q_history[history_idx]
        K_term = self.a * (
            np.roll(Q_delayed, 1, axis=0) + np.roll(Q_delayed, -1, axis=0) +
            np.roll(Q_delayed, 1, axis=1) + np.roll(Q_delayed, -1, axis=1) - 4 * Q_delayed
        )
        P_new = self.P + dt * (-self.gamma * self.P - self.Q + K_term)
        Q_new = self.Q + dt * self.P
        self.P, self.Q = P_new, Q_new
        self.Q_history[t_idx % self.history_steps] = self.Q

    def _calculate_pulse_metrics(self, dt, n_steps):
        """
        C-2 BLOCKER FIX: Measures c_eff along multiple radial angles to compute all isotropy KPIs.
        Also implements quality improvements: hysteresis and proper CI.
        """
        # 1. Initialization
        self.Q.fill(0.0); self.P.fill(0.0); self.Q_history.fill(0.0)
        center = self.grid_size // 2
        
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.Q = np.exp(-((x - center)**2 + (y - center)**2) / 10.0)
        
        # 2. Setup for Radial Front Detection (as per audit)
        num_angles = 16
        thetas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        front_detections = {theta: [] for theta in thetas}
        
        # Recommended Improvement: Hysteresis thresholds for robust detection
        high_thresh, low_thresh = 0.1, 0.07
        is_triggered = {theta: False for theta in thetas}
        max_r_so_far = {theta: 0 for theta in thetas}

        # 3. Simulation and Detection
        for t_idx in range(n_steps):
            self._step_euler(dt, t_idx)
            
            for theta in thetas:
                # Trace a ray of pixels from center to edge
                for r in range(max_r_so_far[theta], center):
                    px = int(center + r * np.cos(theta))
                    py = int(center + r * np.sin(theta))
                    val = self.Q[py, px]
                    
                    if not is_triggered[theta] and val > high_thresh:
                        is_triggered[theta] = True
                    elif is_triggered[theta] and val < low_thresh:
                        is_triggered[theta] = False

                    # Recommended Improvement: Enforce monotone outward progression
                    if is_triggered[theta] and r > max_r_so_far[theta]:
                        max_r_so_far[theta] = r
                
                if max_r_so_far[theta] > 0:
                    front_detections[theta].append((t_idx * dt, max_r_so_far[theta]))
        
        # 4. Calculate c_eff per angle and aggregate metrics
        c_thetas = []
        c_thetas_ci_low, c_thetas_ci_high = [], []

        for theta in thetas:
            detections = np.array(list(dict.fromkeys(front_detections[theta]))) # Keep order, unique
            if len(detections) < 5: continue
            
            times, dists = detections[:, 0], detections[:, 1]
            res = theilslopes(dists, times, 0.95)
            c_thetas.append(res[0]); c_thetas_ci_low.append(res[2]); c_thetas_ci_high.append(res[3])
            
        if not c_thetas:
            return {'ceff_pulse': 0.0, 'ceff_pulse_ic95': 0.0, 'anisotropy_max_pct': 100.0,
                    'var_c_over_c2': 1.0, 'ceff_iso_x': 0.0, 'ceff_iso_y': 0.0, 'ceff_iso_z': 0.0}

        c_thetas = np.array(c_thetas)
        mean_c = np.mean(c_thetas)
        
        # C-2 BLOCKER FIX: Calculate Var(c)/c^2
        var_c_over_c2 = np.var(c_thetas) / (mean_c**2) if mean_c > 0 else 1.0
        
        # C-2: max Δc/c
        anisotropy_max_pct = (np.max(np.abs(c_thetas - mean_c)) / mean_c) * 100 if mean_c > 0 else 100.0
        
        # QUALITY FIX: Use Theil-Sen CI
        ci_width = np.mean(c_thetas_ci_high) - np.mean(c_thetas_ci_low)
        
        c_x = c_thetas[0]; c_y = c_thetas[num_angles // 4]
        
        return {'ceff_pulse': mean_c, 'ceff_pulse_ic95': ci_width / 2.0,
                'anisotropy_max_pct': anisotropy_max_pct, 'var_c_over_c2': var_c_over_c2,
                'ceff_iso_x': c_x, 'ceff_iso_y': c_y, 'ceff_iso_z': np.mean([c_x, c_y])}

    def _calculate_lpc_metrics(self, dt, n_steps):
        """
        C-3 BLOCKER FIX: Implements correct windowing and uses existing spectral_entropy util.
        """
        # 1. Initialization and passive simulation
        self.Q = self.rng.normal(0, 1.0, self.Q.shape); self.P.fill(0.0); self.Q_history.fill(0.0)
        
        center = self.grid_size // 2
        time_series = np.zeros(n_steps)
        for t_idx in range(n_steps):
            self._step_euler(dt, t_idx)
            time_series[t_idx] = self.Q[center, center]

        # 2. Manual windowing and entropy calculation
        win_size, overlap = 2048, 1024
        step = win_size - overlap
        
        if len(time_series) < win_size: return {}, pd.DataFrame()

        block_data = []
        last_K = None
        
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
        lpc_vcount = len(blocks_df)
        if lpc_vcount > 1:
            deltaK_neg_count = (blocks_df['deltaK'][1:] <= 0).sum()
            lpc_deltaK_neg_frac = deltaK_neg_count / (lpc_vcount - 1)
        else:
            lpc_deltaK_neg_frac = 0.0

        return {'lpc_deltaK_neg_frac': lpc_deltaK_neg_frac, 'lpc_vcount': lpc_vcount}, blocks_df

    def run(self):
        dt = 0.1
        pulse_metrics = self._calculate_pulse_metrics(dt, n_steps=3000)
        lpc_metrics, blocks_df = self._calculate_lpc_metrics(dt, n_steps=30000)
        final_run_metrics = {**pulse_metrics, **lpc_metrics}
        return final_run_metrics, blocks_df