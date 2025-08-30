# src/doft/models/model.py
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import math
import warnings

from doft.utils.utils import spectral_entropy

def compute_energy(Q: np.ndarray, P: np.ndarray) -> float:
    """Return total nondimensional energy of the lattice.

    Parameters
    ----------
    Q, P:
        Field displacement and momentum arrays.

    Notes
    -----
    Uses a simple quadratic energy: kinetic (``P``) plus potential (``Q``).
    Coupling terms are omitted; this monitor is intended only to check that
    the semi-implicit scheme with damping does not spuriously increase the
    basic oscillator energy.
    """

    kinetic = 0.5 * np.sum(P ** 2)
    potential = 0.5 * np.sum(Q ** 2)
    return float(kinetic + potential)


def compute_energy_terms(
    Q: np.ndarray,
    P: np.ndarray,
    K: float,
    y_states: np.ndarray | None = None,
    kernel_params: dict | None = None,
) -> dict:
    """Return individual energy contributions of the lattice.

    Parameters
    ----------
    Q, P:
        Field displacement and momentum arrays.
    K:
        Coupling coefficient. When zero the coupling term vanishes.
    y_states:
        Optional array of auxiliary Prony-chain states with shape
        ``(M, *Q.shape)``.
    kernel_params:
        Parameters of the Prony chain. Only the ``"weights"`` array is
        consulted; if absent or empty the memory contribution is skipped.

    Returns
    -------
    dict
        Dictionary with ``kinetic``, ``potential``, ``coupling``, ``memory`` and
        ``total`` contributions.
    """

    kinetic = 0.5 * np.sum(P**2)
    potential = 0.5 * np.sum(Q**2)

    coupling = 0.0
    if K != 0.0:
        grad_x = np.roll(Q, -1, axis=0) - Q
        grad_y = np.roll(Q, -1, axis=1) - Q
        coupling = 0.5 * K * np.sum(grad_x**2 + grad_y**2)

    memory = 0.0
    if y_states is not None and kernel_params:
        weights = np.asarray(kernel_params.get("weights", []), dtype=float)
        if weights.size and y_states.shape[0] == weights.size:
            memory = 0.5 * np.sum(weights[:, None, None] * y_states**2)

    total = kinetic + potential + coupling + memory
    return {
        "kinetic": float(kinetic),
        "potential": float(potential),
        "coupling": float(coupling),
        "memory": float(memory),
        "total": float(total),
    }


def compute_total_energy(
    Q: np.ndarray,
    P: np.ndarray,
    K: float,
    y_states: np.ndarray | None = None,
    kernel_params: dict | None = None,
) -> float:
    """Return lattice energy including coupling and memory contributions."""

    return compute_energy_terms(Q, P, K, y_states, kernel_params)["total"]


class DOFTModel:
    def __init__(
        self,
        grid_size,
        a,
        tau,
        a_ref,
        tau_ref,
        gamma,
        seed,
        boundary_mode: str = "periodic",
        dt_nondim: float | None = None,
        max_pulse_steps: int | None = None,
        max_lpc_steps: int | None = None,
        lpc_duration_physical: float | None = None,
        kernel_params: dict | None = None,
        energy_mode: str = "auto",
        log_steps: bool = False,
        log_path: str | None = None,
        max_ram_bytes: int = 32 * 1024**3,
    ):
        self.grid_size = grid_size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.boundary_mode = boundary_mode

        # STABILITY FIX #1: NONDIMENSIONALIZATION
        # Use reference scales to make all simulation variables of order ~1.
        self.tau_ref = tau_ref  # Reference time scale (e.g., 1.0)
        self.a_ref = a_ref      # Reference coupling scale (e.g., 1.0)

        # Nondimensionalize the parameters for this specific run
        self.a_nondim = a / self.a_ref
        self.tau_nondim = tau / self.tau_ref
        self.gamma_nondim = gamma * self.tau_ref

        # STABILITY FIX #2: SAFE TIME STEP
        # Determine a stable dimensionless time step based on current parameters.
        denom = self.gamma_nondim + abs(self.a_nondim) + 1.0
        if denom > 0:
            gamma_bound = 0.1 / denom
        else:
            gamma_bound = float("inf")
        safe_dt = min(0.02, 0.1, self.tau_nondim / 50.0, gamma_bound)
        if dt_nondim is not None and not math.isclose(dt_nondim, safe_dt, rel_tol=0, abs_tol=1e-12):
            warnings.warn(
                f"Requested dt_nondim={dt_nondim} replaced by stable dt_nondim={safe_dt}",
                RuntimeWarning,
            )
        self.dt_nondim = safe_dt
        self.min_dt_nondim = 1e-6  # Lower bound to prevent infinite halving loops
        self.dt = self.dt_nondim * self.tau_ref  # Actual dt in "physical" units

        # The physical tau is still needed for delay calculation
        self.tau = tau

        self.Q = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.P = np.zeros((grid_size, grid_size), dtype=np.float64)

        # Memory states for Prony-chain kernels (optional)
        self.kernel_params = kernel_params
        if kernel_params and kernel_params.get("weights") is not None:
            n_modes = len(kernel_params.get("weights", []))
            self.y_states = np.zeros((n_modes, grid_size, grid_size), dtype=np.float64)
        else:
            self.y_states = None

        # Delayed state approximated by a single Prony variable
        self.Q_delay = np.zeros((grid_size, grid_size), dtype=np.float64)

        # Keep parameter for compatibility but no longer used to size history buffers
        self.max_ram_bytes = max_ram_bytes

        # Select energy functional
        if energy_mode == "basic":
            self.energy_fn = compute_energy
        elif energy_mode == "total" or (
            energy_mode == "auto" and (self.a_nondim != 0.0 or self.y_states is not None)
        ):
            self.energy_fn = lambda Q, P: compute_total_energy(
                Q, P, self.a_nondim, self.y_states, self.kernel_params
            )
        else:
            self.energy_fn = compute_energy

        # Energy monitoring for source-free simulations
        self.energy_log: list[float] = []
        self.last_energy = self.energy_fn(self.Q, self.P)

        # Track scaling applied to the fields to avoid overflow
        self.scale_threshold = 1e6
        self.scale_accum = 1.0
        self.scale_log: list[float] = []

        # Optional caps for expensive diagnostic runs
        self.max_pulse_steps = max_pulse_steps
        if lpc_duration_physical is not None:
            self.max_lpc_steps = math.ceil(lpc_duration_physical / self.dt)
        else:
            self.max_lpc_steps = max_lpc_steps

        # Optional step-by-step logging
        self.log_steps = log_steps
        self.log_path = log_path or "step_log"
        self.step_log: list[dict] = []
        self._last_K_metric: float | None = None
        self._K_mean = 0.0
        self._K_m2 = 0.0
        self._K_count = 0

    def _get_delayed_q_interpolated(self, t_idx: int | None = None):
        """Return the delayed field stored in the auxiliary state.

        The previous implementation used an explicit history buffer with
        interpolation. It has been replaced by a single auxiliary Prony
        variable updated in :meth:`_step_imex`, so ``t_idx`` is unused but
        retained for backward compatibility.
        """

        return self.Q_delay

    def _laplacian(self, field: np.ndarray, mode: str | None = None) -> np.ndarray:
        """Return discrete Laplacian of ``field`` with boundary ``mode``.

        Parameters
        ----------
        field:
            Array to operate on.
        mode:
            Boundary condition mode. If ``None`` uses ``self.boundary_mode``.
            Supported values are ``"periodic"``, ``"reflective"`` and
            ``"absorbing"``.
        """

        mode = mode or self.boundary_mode

        if mode == "periodic":
            return (
                np.roll(field, 1, axis=0)
                + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1)
                + np.roll(field, -1, axis=1)
                - 4 * field
            )
        if mode == "reflective":
            padded = np.pad(field, 1, mode="edge")
        elif mode == "absorbing":
            padded = np.pad(field, 1, mode="constant", constant_values=0)
        else:
            raise ValueError(f"unknown boundary mode: {mode}")

        return (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4 * field
        )

    def _step_imex(self, t_idx):
        """Advance the state using an IMEX Euler step.

        Linear terms (damping and harmonic restoring force) are treated
        implicitly, while any non-linear contributions are updated explicitly.
        This corresponds to a first-order implicit-explicit (IMEX) Euler scheme
        for the coupled ``(Q, P)`` system.
        """

        Q_prev = self.Q.copy()
        P_prev = self.P.copy()
        energy_prev = self.last_energy

        while True:
            Q_delayed = self._get_delayed_q_interpolated(t_idx)

            # Linear coupling term evaluated explicitly from the delayed field
            K_term = self.a_nondim * self._laplacian(Q_delayed)

            # Placeholder for possible nonlinear contributions (explicit)
            nonlinear_term = 0.0

            # IMEX update: implicit in the linear -Q and -gamma P terms,
            # explicit for K_term and nonlinear_term
            numerator = (
                self.P
                + self.dt_nondim * (K_term + nonlinear_term - self.Q)
            )
            denom = 1.0 + self.dt_nondim * self.gamma_nondim + self.dt_nondim**2
            P_new = numerator / denom
            Q_new = self.Q + self.dt_nondim * P_new

            # Compute norms and rescale if necessary to avoid overflow
            norm_Q = np.linalg.norm(Q_new)
            norm_P = np.linalg.norm(P_new)
            scale = max(norm_Q, norm_P)
            if scale > self.scale_threshold:
                Q_new /= scale
                P_new /= scale
                self.Q_delay /= scale
                self.scale_accum *= scale
                self.last_energy /= scale ** 2
                energy_prev = self.last_energy

            energy_new = self.energy_fn(Q_new, P_new)
            energy_prev_phys = energy_prev * self.scale_accum ** 2
            energy_new_phys = energy_new * self.scale_accum ** 2

            if (
                np.isfinite(P_new).all()
                and np.isfinite(Q_new).all()
                and energy_new_phys <= energy_prev_phys + 1e-12
            ):
                self.P, self.Q = P_new, Q_new
                self.last_energy = energy_new
                alpha = self.dt_nondim / self.tau_nondim if self.tau_nondim > 0 else 0.0
                self.Q_delay = (self.Q_delay + alpha * self.Q) / (1.0 + alpha)
                self.energy_log.append(energy_new_phys)
                self.scale_log.append(self.scale_accum)
                if self.log_steps:
                    self._log_step(t_idx)
                break

            if not (np.isfinite(P_new).all() and np.isfinite(Q_new).all()):
                print(
                    f"WARNING: Non-finite values encountered at step {t_idx}. "
                    f"Reducing dt_nondim from {self.dt_nondim}"
                )
            else:
                print(
                    f"WARNING: Energy increased from {energy_prev} to {energy_new} at step {t_idx}. "
                    f"Reducing dt_nondim from {self.dt_nondim}"
                )

            self.P = P_prev.copy()
            self.Q = Q_prev.copy()
            self.last_energy = energy_prev
            new_dt = self.dt_nondim * 0.5
            if new_dt < self.min_dt_nondim:
                print(
                    f"ERROR: Minimum dt_nondim {self.min_dt_nondim} reached. "
                    "Aborting step."
                )
                self.dt_nondim = self.min_dt_nondim
                self.dt = self.dt_nondim * self.tau_ref
                break

            self.dt_nondim = new_dt
            self.dt = self.dt_nondim * self.tau_ref


    def _log_step(self, t_idx: int):
        """Store per-step energy and LPC metrics if logging is enabled."""

        terms = compute_energy_terms(
            self.Q,
            self.P,
            self.a_nondim,
            self.y_states,
            self.kernel_params,
        )
        K_metric = spectral_entropy(self.Q.flatten())
        if self._last_K_metric is None:
            deltaK = 0.0
        else:
            deltaK = K_metric - self._last_K_metric
        self._last_K_metric = K_metric

        self._K_count += 1
        delta = K_metric - self._K_mean
        self._K_mean += delta / self._K_count
        self._K_m2 += delta * (K_metric - self._K_mean)
        K_var = self._K_m2 / (self._K_count - 1) if self._K_count > 1 else 0.0

        self.step_log.append(
            {
                "step": t_idx,
                "kinetic": terms["kinetic"],
                "potential": terms["potential"],
                "coupling": terms["coupling"],
                "memory": terms["memory"],
                "K_metric": K_metric,
                "lpc_slope": deltaK,
                "lpc_var": K_var,
            }
        )

    def save_step_log(self):
        """Persist step logs to CSV and JSON files."""

        if not self.log_steps or not self.step_log:
            return
        df = pd.DataFrame(self.step_log)
        csv_path = f"{self.log_path}.csv"
        json_path = f"{self.log_path}.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records")
    
    def _calculate_pulse_metrics(self, n_steps, noise_std: float = 0.0):
        r"""Estimate wave-front speed using multiple noise-relative thresholds.

        Parameters
        ----------
        n_steps:
            Number of integration steps.
        noise_std:
            Standard deviation of the synthetic noise added before injecting the
            pulse. The resulting floor :math:`\xi` sets the detection thresholds
            ``{1σ, 3σ, 5σ}``.
        """

        # Reset fields and optional pre-pulse noise
        self.Q.fill(0.0)
        self.P.fill(0.0)
        self.Q_delay.fill(0.0)
        if noise_std > 0.0:
            self.Q += self.rng.normal(0.0, noise_std, size=self.Q.shape)

        # Noise floor and thresholds relative to it
        xi_floor = float(np.std(self.Q))
        xi_floor = max(xi_floor, 1e-12)
        thresholds = xi_floor * np.array([1.0, 3.0, 5.0])

        center = self.grid_size // 2

        # Inject Gaussian pulse
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.Q += 0.1 * np.exp(-((x - center) ** 2 + (y - center) ** 2) / 10.0)
        # Update stored energy after pulse injection so the stability guard
        # does not interpret the added pulse energy as a spurious increase.
        self.last_energy = self.energy_fn(self.Q, self.P)

        num_angles = 16
        thetas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

        front_detections = {
            (theta, thr_idx): []
            for theta in thetas
            for thr_idx in range(len(thresholds))
        }
        max_r_so_far = {
            (theta, thr_idx): 0
            for theta in thetas
            for thr_idx in range(len(thresholds))
        }

        for t_idx in range(n_steps):
            self._step_imex(t_idx)
            t_now = t_idx * self.dt
            for theta in thetas:
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                for thr_idx, thr in enumerate(thresholds):
                    r_start = max_r_so_far[(theta, thr_idx)]
                    for r in range(r_start, center):
                        px = int(center + r * cos_t)
                        py = int(center + r * sin_t)
                        if self.Q[py, px] > thr:
                            max_r_so_far[(theta, thr_idx)] = r
                    rmax = max_r_so_far[(theta, thr_idx)]
                    if rmax > 0:
                        front_detections[(theta, thr_idx)].append((t_now, rmax))

        c_thetas = []
        c_thetas_ci_low = []
        c_thetas_ci_high = []

        # Collect per-threshold speeds for later aggregation/inspection
        c_by_thr = {thr_idx: [] for thr_idx in range(len(thresholds))}
        ci_low_by_thr = {thr_idx: [] for thr_idx in range(len(thresholds))}
        ci_high_by_thr = {thr_idx: [] for thr_idx in range(len(thresholds))}

        for theta in thetas:
            c_thr = []
            ci_lo_thr = []
            ci_hi_thr = []
            for thr_idx in range(len(thresholds)):
                detections = np.array(
                    list(dict.fromkeys(front_detections[(theta, thr_idx)]))
                )
                if len(detections) < 10:
                    continue
                times, dists = detections[:, 0], detections[:, 1]
                res = theilslopes(dists, times, 0.95)
                c_thr.append(res[0])
                ci_lo_thr.append(res[2])
                ci_hi_thr.append(res[3])
                c_by_thr[thr_idx].append(res[0])
                ci_low_by_thr[thr_idx].append(res[2])
                ci_high_by_thr[thr_idx].append(res[3])
            if c_thr:
                c_thetas.append(float(np.mean(c_thr)))
                c_thetas_ci_low.append(float(np.mean(ci_lo_thr)))
                c_thetas_ci_high.append(float(np.mean(ci_hi_thr)))

        if not c_thetas:
            return {
                'xi_floor': xi_floor,
                'ceff_pulse': 0.0,
                'ceff_pulse_ic95_lo': 0.0,
                'ceff_pulse_ic95_hi': 0.0,
                'anisotropy_max_pct': 100.0,
                'var_c_over_c2': 1.0,
                'ceff_iso_x': 0.0,
                'ceff_iso_y': 0.0,
                'ceff_iso_z': 0.0,
                'ceff_iso_diag': 0.0,
                'ceff_pulse_by_thr': [0.0 for _ in thresholds],
            }

        c_thetas = np.array(c_thetas)
        mean_c = float(np.mean(c_thetas))
        var_c_over_c2 = np.var(c_thetas) / (mean_c ** 2) if mean_c > 0 else 1.0
        anisotropy_max_pct = (
            np.max(np.abs(c_thetas - mean_c)) / mean_c * 100 if mean_c > 0 else 100.0
        )
        ci_low = float(np.mean(c_thetas_ci_low))
        ci_high = float(np.mean(c_thetas_ci_high))

        c_x = c_thetas[0] if len(c_thetas) > 0 else 0.0
        c_y = c_thetas[num_angles // 4] if len(c_thetas) > num_angles // 4 else 0.0
        c_z = c_thetas[num_angles // 8] if len(c_thetas) > num_angles // 8 else 0.0
        c_diag = (
            c_thetas[3 * num_angles // 8]
            if len(c_thetas) > 3 * num_angles // 8
            else 0.0
        )

        c_by_thr_list = [
            float(np.mean(c_by_thr[i])) if c_by_thr[i] else 0.0
            for i in range(len(thresholds))
        ]

        return {
            'xi_floor': xi_floor,
            'ceff_pulse': mean_c,
            'ceff_pulse_ic95_lo': ci_low,
            'ceff_pulse_ic95_hi': ci_high,
            'anisotropy_max_pct': anisotropy_max_pct,
            'var_c_over_c2': var_c_over_c2,
            'ceff_iso_x': c_x,
            'ceff_iso_y': c_y,
            'ceff_iso_z': c_z,
            'ceff_iso_diag': c_diag,
            'ceff_pulse_by_thr': c_by_thr_list,
        }

    def _calculate_lpc_metrics(self, n_steps):
        self.Q = self.rng.normal(0, 0.1, self.Q.shape); self.P.fill(0.0); self.Q_delay.fill(0.0)
        center = self.grid_size // 2
        time_series = np.zeros(n_steps)
        for t_idx in range(n_steps):
            self._step_imex(t_idx)
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
                block_data.append({'window_id': i,
                                    'K_metric': np.nan,
                                    'deltaK': np.nan,
                                    'block_skipped': 1})
                continue
            K_metric = spectral_entropy(window_data)
            deltaK = K_metric - last_K if last_K is not None else 0.0
            block_data.append({'window_id': i,
                                'K_metric': K_metric,
                                'deltaK': deltaK,
                                'block_skipped': 0})
            last_K = K_metric

        blocks_df = pd.DataFrame(block_data)

        valid_blocks = blocks_df[blocks_df['block_skipped'] == 0]
        windows_analyzed = len(valid_blocks)
        if windows_analyzed > 1:
            deltaK_neg_count = (valid_blocks['deltaK'][1:] <= 0).sum()
            lpc_ok_frac = deltaK_neg_count / (windows_analyzed - 1)
        else:
            lpc_ok_frac = 0.0

        return {
            'lpc_ok_frac': lpc_ok_frac,
            'lpc_vcount': 0,
            'lpc_windows_analyzed': windows_analyzed,
            'block_skipped': block_skipped,
        }, blocks_df

    def run(self):
        # Adjust n_steps to account for the much smaller dt, simulating a similar physical duration.
        # old_dt=0.1, new_dt=0.005*tau_ref. Ratio is ~20.
        pulse_steps = int(3000 * (0.1 / self.dt))
        if self.max_pulse_steps is not None:
            pulse_steps = min(pulse_steps, self.max_pulse_steps)
        lpc_steps = int(30000 * (0.1 / self.dt))
        if self.max_lpc_steps is not None:
            lpc_steps = min(lpc_steps, self.max_lpc_steps)

        pulse_metrics = self._calculate_pulse_metrics(n_steps=pulse_steps)
        lpc_metrics, blocks_df = self._calculate_lpc_metrics(n_steps=lpc_steps)
        final_run_metrics = {**pulse_metrics, **lpc_metrics}
        if self.log_steps:
            self.save_step_log()
        return final_run_metrics, blocks_df
    