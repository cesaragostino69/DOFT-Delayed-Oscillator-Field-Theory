# src/doft/models/model.py
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import math

from doft.utils.utils import spectral_entropy

class DOFTModel:
    def __init__(self, grid_size, a, tau, a_ref, tau_ref, gamma, seed, dt_nondim: float = 0.005):
        self.grid_size = grid_size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        # STABILITY FIX #1: NONDIMENSIONALIZATION
        # Use reference scales to make all simulation variables of order ~1.
        self.tau_ref = tau_ref  # Reference time scale (e.g., 1.0)
        self.a_ref = a_ref      # Reference coupling scale (e.g., 1.0)

        # STABILITY FIX #2: SAFE TIME STEP
        # Use a small, dimensionless time step as recommended by auditors.
        self.dt_nondim = dt_nondim
        self.dt = self.dt_nondim * self.tau_ref  # Actual dt in "physical" units

        # Nondimensionalize the parameters for this specific run
        self.a_nondim = a / self.a_ref
        self.tau_nondim = tau / self.tau_ref
        self.gamma_nondim = gamma * self.tau_ref

        # The physical tau is still needed for delay calculation
        self.tau = tau

        # Precompute delay in time steps to avoid repeated calculation
        self.delay_in_steps = self.tau / self.dt

        # Correctly size the history buffer with the new, smaller dt
        self.history_steps = int(math.ceil(self.tau / self.dt)) + 5 # Added safe margin

        self.Q = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.P = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.Q_history = np.zeros((self.history_steps, grid_size, grid_size), dtype=np.float64)

    def _get_delayed_q_interpolated(self, t_idx):
        """
        STABILITY FIX #3: LINEAR INTERPOLATION FOR DELAYS
        Improves accuracy and stability by interpolating between two past time steps
        instead of taking the nearest neighbor.
        """
        delay_in_steps = self.delay_in_steps

        idx_floor = int(math.floor(delay_in_steps))
        idx_ceil = int(math.ceil(delay_in_steps))

        if idx_floor == idx_ceil:
            # The delay is an exact multiple of dt
            history_idx = (t_idx - idx_floor) % self.history_steps
            return self.Q_history[history_idx]

        # Get the two bracketing time steps from history
        hist_idx1 = (t_idx - idx_floor) % self.history_steps
        hist_idx2 = (t_idx - idx_ceil) % self.history_steps
        Q1 = self.Q_history[hist_idx1]
        Q2 = self.Q_history[hist_idx2]

        # Interpolation factor (fractional part)
        frac = delay_in_steps - idx_floor

        # Linearly interpolate between the two states
        return Q2 * frac + Q1 * (1.0 - frac)

    def _step_euler(self, t_idx):
        Q_delayed = self._get_delayed_q_interpolated(t_idx)

        # The equations of motion now use dimensionless parameters
        K_term = self.a_nondim * (
            np.roll(Q_delayed, 1, axis=0) + np.roll(Q_delayed, -1, axis=0) +
            np.roll(Q_delayed, 1, axis=1) + np.roll(Q_delayed, -1, axis=1) - 4 * Q_delayed
        )
        # We use the dimensionless dt for the update step
        P_new = self.P + self.dt_nondim * (-self.gamma_nondim * self.P - self.Q + K_term)
        Q_new = self.Q + self.dt_nondim * self.P

        self.P, self.Q = P_new, Q_new
        self.Q_history[t_idx % self.history_steps] = self.Q

    def _calculate_pulse_metrics(self, n_steps):
        self.Q.fill(0.0); self.P.fill(0.0); self.Q_history.fill(0.0)
        center = self.grid_size // 2

        # Use amplitudes of Order(1) for stability
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.Q = 0.1 * np.exp(-((x - center)**2 + (y - center)**2) / 10.0)

        num_angles = 16
        thetas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        front_detections = {theta: [] for theta in thetas}

        high_thresh, low_thresh = 0.01, 0.007 # Reduced thresholds for smaller amplitude
        is_triggered = {theta: False for theta in thetas}
        max_r_so_far = {theta: 0 for theta in thetas}

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
                    # Distances and times are in physical (not dimensionless) units for KPI
                    front_detections[theta].append((t_idx * self.dt, max_r_so_far[theta]))

        c_thetas = []; c_thetas_ci_low, c_thetas_ci_high = [], []
        for theta in thetas:
            detections = np.array(list(dict.fromkeys(front_detections[theta])))
            if len(detections) < 10: continue # Need more points with small dt
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
        c_x = c_thetas[0]; c_y = c_thetas[num_angles // 4]; c_z = c_thetas[num_angles // 8]
        return {'ceff_pulse': mean_c, 'ceff_pulse_ic95': ci_width / 2.0,
                'anisotropy_max_pct': anisotropy_max_pct, 'var_c_over_c2': var_c_over_c2,
                'ceff_iso_x': c_x, 'ceff_iso_y': c_y, 'ceff_iso_z': c_z}

    def _calculate_lpc_metrics(self, n_steps):
        self.Q = self.rng.normal(0, 0.1, self.Q.shape); self.P.fill(0.0); self.Q_history.fill(0.0)
        center = self.grid_size // 2
        time_series = np.zeros(n_steps)
        for t_idx in range(n_steps):
            self._step_euler(t_idx)
            time_series[t_idx] = self.Q[center, center]

        # STABILITY FIX #4: NUMERICAL GUARD
        # Check for non-finite values before spectral calculations.
        # We no longer abort the entire metric calculation if non-finite
        # values appear; instead, individual windows will be skipped.
        if not np.all(np.isfinite(time_series)):
            print("  WARNING: Non-finite values detected in time series.")

        win_size, overlap = 4096, 2048 # Larger window for finer frequency resolution with small dt
        step = win_size - overlap
        if len(time_series) < win_size:
            return {'block_skipped': 0}, pd.DataFrame()

        block_data, last_K = [], None
        block_skipped = 0
        num_windows = (len(time_series) - win_size) // step + 1
        for i in range(num_windows):
            window_data = time_series[i*step : i*step + win_size]
            if not np.isfinite(window_data).all():
                block_skipped += 1
                continue
            K_metric = spectral_entropy(window_data)
            deltaK = K_metric - last_K if last_K is not None else 0.0
            block_data.append({'window_id': i, 'K_metric': K_metric, 'deltaK': deltaK})
            last_K = K_metric
        if not block_data:
            return {'block_skipped': block_skipped}, pd.DataFrame()
        blocks_df = pd.DataFrame(block_data)

        windows_analyzed = len(blocks_df)
        if windows_analyzed > 1:
            deltaK_neg_count = (blocks_df['deltaK'][1:] <= 0).sum()
            lpc_deltaK_neg_frac = deltaK_neg_count / (windows_analyzed - 1)
        else:
            lpc_deltaK_neg_frac = 0.0

        return {'lpc_deltaK_neg_frac': lpc_deltaK_neg_frac,
                'lpc_brake_count': 0,
                'lpc_windows_analyzed': windows_analyzed,
                'block_skipped': block_skipped}, blocks_df

    def run(self):
        # Adjust n_steps to account for the much smaller dt, simulating a similar physical duration.
        # old_dt=0.1, new_dt=0.005*tau_ref. Ratio is ~20.
        pulse_steps = int(3000 * (0.1 / self.dt))
        lpc_steps = int(30000 * (0.1 / self.dt))

        pulse_metrics = self._calculate_pulse_metrics(n_steps=pulse_steps)
        lpc_metrics, blocks_df = self._calculate_lpc_metrics(n_steps=lpc_steps)
        final_run_metrics = {**pulse_metrics, **lpc_metrics}
        return final_run_metrics, blocks_df
