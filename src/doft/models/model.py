# src/doft/models/model.py
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, entropy
from scipy.signal import welch, detrend

class DOFTModel:
    """
    Implements the physical simulation of the Delayed Oscillator Field Theory (DOFT).
    This version is aligned with the requirements of the Phase 1 counter-trial.
    """
    def __init__(self, grid_size, a, tau, gamma, seed):
        self.grid_size = grid_size
        self.a = a          # Coupling parameter
        self.tau = tau      # Delay parameter
        self.gamma = gamma  # Damping parameter (NOW IN USE!)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
        # Initialize the Q (position) and P (momentum) fields
        self.Q = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.P = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # History to handle time delays
        self.history_steps = int(np.ceil(self.tau * 2)) + 1
        self.Q_history = np.zeros((self.history_steps, grid_size, grid_size), dtype=np.float64)

    def _step_euler(self, dt, t_idx):
        """
        Advances the simulation by one time step using the Euler method.
        """
        # --- GET THE DELAYED COUPLING TERM ---
        # Calculate the index in the history corresponding to (t - tau)
        history_idx = (t_idx - int(self.tau / dt)) % self.history_steps
        Q_delayed = self.Q_history[history_idx]
        
        # The coupling is the difference with neighbors (discrete Laplacian)
        # We use np.roll for periodic boundary conditions (a torus)
        K_term = self.a * (
            np.roll(Q_delayed, 1, axis=0) + np.roll(Q_delayed, -1, axis=0) +
            np.roll(Q_delayed, 1, axis=1) + np.roll(Q_delayed, -1, axis=1) - 4 * Q_delayed
        )
        
        # --- EQUATIONS OF MOTION (UPDATED) ---
        # CRITICAL FIX! The damping term -gamma * P is added
        P_new = self.P + dt * (-self.gamma * self.P - self.Q + K_term)
        Q_new = self.Q + dt * self.P
        
        self.P, self.Q = P_new, Q_new
        
        # Save the current state to the history
        self.Q_history[t_idx % self.history_steps] = self.Q

    def _calculate_pulse_metrics(self, dt, n_steps):
        """
        Experiment A: Measures emergent c_eff and isotropy.
        """
        # 1. Initialization for the pulse experiment
        self.Q.fill(0.0)
        self.P.fill(0.0)
        self.Q_history.fill(0.0)
        center = self.grid_size // 2
        
        # Inject an initial Gaussian pulse at the center
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        dist_sq = (x - center)**2 + (y - center)**2
        self.Q = np.exp(-dist_sq / 10.0)
        
        # 2. Simulation and front detection
        front_detections = {'x': [], 'y': []} # We store (time, distance)
        detection_threshold = 0.1
        
        for t_idx in range(n_steps):
            self._step_euler(dt, t_idx)
            # Detect the front along the x and y axes
            for axis_idx, axis_name in [(0, 'y'), (1, 'x')]:
                line = self.Q[center, :] if axis_name == 'x' else self.Q[:, center]
                crossings = np.where(line > detection_threshold)[0]
                if len(crossings) > 0:
                    max_dist = np.abs(crossings - center).max()
                    if max_dist > 0:
                        front_detections[axis_name].append((t_idx * dt, max_dist))

        # 3. Calculation of c_eff with Theil-Sen (robust to outliers)
        results = {}
        ceff_values = []
        for axis in ['x', 'y']:
            detections = np.array(list(set(front_detections[axis])))
            if len(detections) < 5:
                # Not enough points for a reliable fit
                results[f'ceff_iso_{axis}'] = np.nan
                continue
            
            times, dists = detections[:, 0], detections[:, 1]
            # res = (slope, intercept, lo_slope, hi_slope) where slope is c_eff
            res = theilslopes(dists, times)
            ceff = res[0]
            ceff_values.append(ceff)
            results[f'ceff_iso_{axis}'] = ceff
        
        # For Z, in a 2D simulation, we assume it's equal to the average of X and Y
        if ceff_values:
            results['ceff_iso_z'] = np.mean(ceff_values)
            ceff_values.append(results['ceff_iso_z'])
            
            # Main metric: ceff_pulse and its 95% CI
            mean_ceff = np.mean(ceff_values)
            # The Theil-Sen CI is for the slope. Here we use a simple std for the CI.
            std_ceff = np.std(ceff_values)
            results['ceff_pulse'] = mean_ceff
            results['ceff_pulse_ic95'] = 1.96 * std_ceff # Approximation
            
            # Isotropy metric (C-2)
            max_dev = np.max(np.abs(ceff_values - mean_ceff))
            results['anisotropy_max_pct'] = (max_dev / mean_ceff) * 100 if mean_ceff > 0 else 0.0
        else: # If no pulse was detected
            results['ceff_pulse'] = 0.0
            results['ceff_pulse_ic95'] = 0.0
            results['anisotropy_max_pct'] = 100.0

        return results
        
    def _calculate_lpc_metrics(self, dt, n_steps):
        """
        Experiment B: Validates the Law of Preservation of Chaos (LPC).
        """
        # 1. Initialization: high-amplitude Gaussian noise
        self.Q = self.rng.normal(0, 1.0, self.Q.shape)
        self.P = self.rng.normal(0, 1.0, self.P.shape)
        self.Q_history.fill(0.0)
        
        # 2. Simulation in passive mode
        # We use the time series of the central oscillator as a representative sample
        center = self.grid_size // 2
        time_series = np.zeros(n_steps)
        for t_idx in range(n_steps):
            self._step_euler(dt, t_idx)
            time_series[t_idx] = self.Q[center, center]

        # 3. Spectral entropy analysis by windows
        win_size = 2048
        overlap = win_size // 2
        if len(time_series) < win_size:
            # Not enough data
            return {}, pd.DataFrame() 
            
        # welch gives us the Power Spectral Density (PSD) by windows
        freqs, psd = welch(detrend(time_series), fs=1/dt, nperseg=win_size, noverlap=overlap)
        
        block_data = []
        last_K = None
        for i in range(psd.shape[1]):
            psd_window = psd[:, i]
            # Normalize the PSD so it sums to 1
            psd_norm = psd_window / psd_window.sum()
            K_metric = entropy(psd_norm) # Shannon entropy
            
            deltaK = K_metric - last_K if last_K is not None else 0.0
            block_data.append({'window_id': i, 'K_metric': K_metric, 'deltaK': deltaK})
            last_K = K_metric
            
        blocks_df = pd.DataFrame(block_data)
        
        # 4. Calculate summary metrics for runs.csv (C-3)
        lpc_vcount = len(blocks_df)
        if lpc_vcount > 1:
            # We ignore the first window for deltaK calculation
            deltaK_neg_count = (blocks_df['deltaK'][1:] <= 0).sum()
            lpc_deltaK_neg_frac = deltaK_neg_count / (lpc_vcount - 1)
        else:
            lpc_deltaK_neg_frac = 0.0

        lpc_metrics = {
            'lpc_deltaK_neg_frac': lpc_deltaK_neg_frac,
            'lpc_vcount': lpc_vcount
        }
        
        return lpc_metrics, blocks_df

    def run(self):
        """
        Executes the Phase 1 experiment sequence and returns all metrics.
        """
        dt = 0.1
        
        # --- Execute Experiment A: Pulse and Isotropy ---
        pulse_steps = 2000
        pulse_metrics = self._calculate_pulse_metrics(dt, pulse_steps)
        
        # --- Execute Experiment B: LPC ---
        lpc_steps = 30000
        lpc_metrics, blocks_df = self._calculate_lpc_metrics(dt, lpc_steps)
        
        # --- Combine results ---
        final_run_metrics = {**pulse_metrics, **lpc_metrics}
        
        return final_run_metrics, blocks_df