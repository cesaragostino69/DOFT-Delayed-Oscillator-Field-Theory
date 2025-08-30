import logging
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from doft.utils.utils import spectral_entropy

DEFAULT_MAX_RAM_BYTES = 32 * 1024 ** 3

logger = logging.getLogger(__name__)


def compute_energy(Q: np.ndarray, P: np.ndarray) -> float:
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
    kinetic = 0.5 * np.sum(P ** 2)
    potential = 0.5 * np.sum(Q ** 2)

    coupling = 0.0
    if K != 0.0:
        grad_x = np.roll(Q, -1, axis=0) - Q
        grad_y = np.roll(Q, -1, axis=1) - Q
        coupling = 0.5 * K * np.sum(grad_x ** 2 + grad_y ** 2)

    memory = 0.0
    if y_states is not None and kernel_params:
        weights = np.asarray(kernel_params.get("weights", []), dtype=float)
        if weights.size and y_states.shape[0] == weights.size:
            memory = 0.5 * np.sum(weights[:, None, None] * y_states ** 2)

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
    return compute_energy_terms(Q, P, K, y_states, kernel_params)["total"]


class DOFTModel:
    """Simplified DOFT model using an IMEX/Leapfrog integrator with Prony memory."""

    def __init__(
        self,
        grid_size: int,
        a: float,
        tau: float,
        a_ref: float,
        tau_ref: float,
        gamma: float,
        seed: int,
        boundary_mode: str = "periodic",
        dt_nondim: float | None = None,
        max_pulse_steps: int | None = None,
        max_lpc_steps: int | None = None,
        lpc_duration_physical: float | None = None,
        kernel_params: dict | None = None,
        energy_mode: str = "auto",
        log_steps: bool = False,
        log_path: str | None = None,
<<<<<<< ours
        max_ram_bytes: int | None = None,
    ) -> None:
=======
    ):
        """Create a new model instance.

        Parameters
        ----------
        grid_size, a, tau, a_ref, tau_ref, gamma, seed :
            Standard model configuration parameters.
        boundary_mode : str, optional
            Boundary condition for the lattice (default ``"periodic"``).
        dt_nondim : float, optional
            Desired nondimensional time step. If unstable a safe value is
            substituted.
        max_pulse_steps : int, optional
            Optional cap on the number of steps used for pulse metrics.
        max_lpc_steps : int, optional
            Optional cap on steps for LPC metrics. Ignored if
            ``lpc_duration_physical`` is provided.
        lpc_duration_physical : float, optional
            Physical duration for the LPC analysis. Converted to steps using
            ``dt`` and stored in ``max_lpc_steps``.
        kernel_params, energy_mode, log_steps, log_path :
            Additional configuration options.
        """

>>>>>>> theirs
        self.grid_size = grid_size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.boundary_mode = boundary_mode

        # nondimensionalisation
        self.tau_ref = tau_ref
        self.a_ref = a_ref
        self.a_nondim = a / self.a_ref
        self.tau_nondim = tau / self.tau_ref
        self.gamma_nondim = gamma * self.tau_ref

        denom = self.gamma_nondim + abs(self.a_nondim) + 1.0
        gamma_bound = 0.1 / denom if denom > 0 else float("inf")
        safe_dt = min(0.02, 0.1, self.tau_nondim / 50.0, gamma_bound)
        if dt_nondim is not None and not math.isclose(dt_nondim, safe_dt, rel_tol=0, abs_tol=1e-12):
            warnings.warn(
                f"Requested dt_nondim={dt_nondim} replaced by stable dt_nondim={safe_dt}",
                RuntimeWarning,
            )
        self.dt_nondim = safe_dt
        self.min_dt_nondim = 1e-6
        self.dt = self.dt_nondim * self.tau_ref
        self.tau = tau

        self.Q = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.P = np.zeros((grid_size, grid_size), dtype=np.float64)

        # Prony-chain memory states
        self.kernel_params = kernel_params
        if kernel_params and kernel_params.get("weights") is not None:
            n_modes = len(kernel_params.get("weights", []))
            self.y_states = np.zeros((n_modes, grid_size, grid_size), dtype=np.float64)
            self.y_thetas = np.asarray(kernel_params.get("thetas", np.zeros(n_modes)), dtype=float)
        else:
            self.y_states = None
            self.y_thetas = None

        self.max_ram_bytes = max_ram_bytes or DEFAULT_MAX_RAM_BYTES

        bytes_per_slice = grid_size * grid_size * np.dtype(np.float64).itemsize
        memory_used = 2 * bytes_per_slice
        if self.y_states is not None:
            memory_used += self.y_states.shape[0] * bytes_per_slice
        if memory_used > self.max_ram_bytes:
            raise MemoryError(
                f"Requested configuration needs {memory_used / 1024 ** 3:.2f} GB, exceeds limit {self.max_ram_bytes / 1024 ** 3:.2f} GB"
            )
        if memory_used > 0.8 * self.max_ram_bytes:
            warnings.warn(
                f"Estimated memory usage {memory_used / 1024 ** 3:.2f} GB approaching limit {self.max_ram_bytes / 1024 ** 3:.2f} GB",
                ResourceWarning,
            )

        # Select energy functional
        if energy_mode == "basic":
            self.energy_fn = compute_energy
        elif energy_mode == "total" or (energy_mode == "auto" and (self.a_nondim != 0.0 or self.y_states is not None)):
            self.energy_fn = lambda Q, P: compute_total_energy(Q, P, self.a_nondim, self.y_states, self.kernel_params)
        else:
            self.energy_fn = compute_energy

        self.energy_log: list[float] = []
        self.last_energy = self.energy_fn(self.Q, self.P)

        self.scale_threshold = 1e6
        self.scale_accum = 1.0
        self.scale_log: list[float] = []

        self.max_pulse_steps = max_pulse_steps
        self.max_lpc_steps = max_lpc_steps
        if lpc_duration_physical is not None:
            self.max_lpc_steps = int(math.ceil(lpc_duration_physical / self.dt))

        self.log_steps = log_steps
        self.log_path = log_path or "step_log"
        self.step_log: list[dict] = []
        self._last_K_metric: float | None = None
        self._K_mean = 0.0
        self._K_m2 = 0.0
        self._K_count = 0

    def _laplacian(self, field: np.ndarray, mode: str | None = None) -> np.ndarray:
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

    def _step_imex(self, t_idx: int) -> None:
        Q_prev = self.Q.copy()
        P_prev = self.P.copy()
        energy_prev = self.last_energy

        while True:
            lap = self.a_nondim * self._laplacian(self.Q)
            memory_force = 0.0
            if self.y_states is not None:
                memory_force = np.sum(self.y_states, axis=0)

            P_new = (
                self.P + self.dt_nondim * (-self.Q + lap + memory_force)
            ) / (1.0 + self.dt_nondim * self.gamma_nondim)
            Q_new = self.Q + self.dt_nondim * P_new

            if self.y_states is not None:
                weights = np.asarray(self.kernel_params.get("weights", []), dtype=float)
                self.y_states += self.dt_nondim * (
                    self.y_thetas[:, None, None] * self.y_states + weights[:, None, None] * P_new
                )

            norm_Q = np.linalg.norm(Q_new)
            norm_P = np.linalg.norm(P_new)
            scale = max(norm_Q, norm_P)
            if scale > self.scale_threshold:
                Q_new /= scale
                P_new /= scale
                if self.y_states is not None:
                    self.y_states /= scale
                self.scale_accum *= scale
                self.last_energy /= scale ** 2
                energy_prev = self.last_energy

            energy_new = self.energy_fn(Q_new, P_new)
            if (
                np.isfinite(P_new).all()
                and np.isfinite(Q_new).all()
                and energy_new <= energy_prev + 1e-12
            ):
                self.P, self.Q = P_new, Q_new
                self.last_energy = energy_new
                self.energy_log.append(energy_new * self.scale_accum ** 2)
                self.scale_log.append(self.scale_accum)
                if self.log_steps:
                    self._log_step(t_idx)
                break

            self.P = P_prev.copy()
            self.Q = Q_prev.copy()
            self.last_energy = energy_prev
            new_dt = self.dt_nondim * 0.5
            if new_dt < self.min_dt_nondim:
                self.dt_nondim = self.min_dt_nondim
                self.dt = self.dt_nondim * self.tau_ref
                break
            self.dt_nondim = new_dt
            self.dt = self.dt_nondim * self.tau_ref

    def _log_step(self, t_idx: int) -> None:
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

    def save_step_log(self) -> None:
        if not self.log_steps or not self.step_log:
            return
        df = pd.DataFrame(self.step_log)
        df.to_csv(f"{self.log_path}.csv", index=False)
        df.to_json(f"{self.log_path}.json", orient="records")

    def _calculate_pulse_metrics(self, n_steps: int, noise_std: float = 0.0) -> dict:
        self.Q.fill(0.0)
        self.P.fill(0.0)
        if self.y_states is not None:
            self.y_states.fill(0.0)
        if noise_std > 0.0:
            self.Q += self.rng.normal(0.0, noise_std, size=self.Q.shape)

        xi_floor = float(np.std(self.Q))
        xi_floor = max(xi_floor, 1e-12)
        thresholds = xi_floor * np.array([1.0, 3.0, 5.0])
        center = self.grid_size // 2
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.Q += 0.1 * np.exp(-((x - center) ** 2 + (y - center) ** 2) / 10.0)
        self.last_energy = self.energy_fn(self.Q, self.P)

        thetas = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        front_detections = {(theta, idx): [] for theta in thetas for idx in range(len(thresholds))}
        max_r_so_far = {(theta, idx): 0 for theta in thetas for idx in range(len(thresholds))}

        for t_idx in range(n_steps):
            self._step_imex(t_idx)
            t_now = t_idx * self.dt
            for theta in thetas:
                ct, st = np.cos(theta), np.sin(theta)
                for thr_idx, thr in enumerate(thresholds):
                    r_start = max_r_so_far[(theta, thr_idx)]
                    for r in range(r_start, center):
                        px = int(center + r * ct)
                        py = int(center + r * st)
                        if self.Q[py, px] > thr:
                            max_r_so_far[(theta, thr_idx)] = r
                    rmax = max_r_so_far[(theta, thr_idx)]
                    if rmax > 0:
                        front_detections[(theta, thr_idx)].append((t_now, rmax))

        c_thetas = []
        c_thetas_ci_low = []
        c_thetas_ci_high = []
        c_by_thr = {thr_idx: [] for thr_idx in range(len(thresholds))}
        for theta in thetas:
            for thr_idx in range(len(thresholds)):
                detections = front_detections[(theta, thr_idx)]
                if len(detections) < 2:
                    continue
                times, dists = zip(*detections)
                slope, intercept, lo, hi = theilslopes(dists, times, 0.95)
                c_thetas.append(slope)
                c_thetas_ci_low.append(lo)
                c_thetas_ci_high.append(hi)
                c_by_thr[thr_idx].append(slope)
        if not c_thetas:
            logger.warning(
                "No wavefronts detected during pulse metrics calculation; "
                "consider revising parameters."
            )
            return {
                "xi_floor": xi_floor,
                "ceff_pulse": 0.0,
                "ceff_pulse_ic95_lo": 0.0,
                "ceff_pulse_ic95_hi": 0.0,
                "anisotropy_max_pct": 100.0,
                "var_c_over_c2": 1.0,
                "ceff_iso_x": 0.0,
                "ceff_iso_y": 0.0,
                "ceff_iso_z": 0.0,
                "ceff_iso_diag": 0.0,
                "ceff_pulse_by_thr": [0.0 for _ in thresholds],
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
        c_y = c_thetas[len(thetas) // 4] if len(c_thetas) > len(thetas) // 4 else 0.0
        c_z = c_thetas[len(thetas) // 8] if len(c_thetas) > len(thetas) // 8 else 0.0
        c_diag = c_thetas[3 * len(thetas) // 8] if len(c_thetas) > 3 * len(thetas) // 8 else 0.0
        c_by_thr_list = [float(np.mean(c_by_thr[i])) if c_by_thr[i] else 0.0 for i in range(len(thresholds))]
        return {
            "xi_floor": xi_floor,
            "ceff_pulse": mean_c,
            "ceff_pulse_ic95_lo": ci_low,
            "ceff_pulse_ic95_hi": ci_high,
            "anisotropy_max_pct": anisotropy_max_pct,
            "var_c_over_c2": var_c_over_c2,
            "ceff_iso_x": c_x,
            "ceff_iso_y": c_y,
            "ceff_iso_z": c_z,
            "ceff_iso_diag": c_diag,
            "ceff_pulse_by_thr": c_by_thr_list,
        }

    def _calculate_lpc_metrics(self, n_steps: int):
        self.Q = self.rng.normal(0, 0.1, self.Q.shape)
        self.P.fill(0.0)
        if self.y_states is not None:
            self.y_states.fill(0.0)
        center = self.grid_size // 2
        time_series = np.zeros(n_steps)
        for t_idx in range(n_steps):
            self._step_imex(t_idx)
            time_series[t_idx] = self.Q[center, center]
        if not np.all(np.isfinite(time_series)):
            print("  WARNING: Non-finite values detected in time series.")
        win_size, overlap = 4096, 2048
        step = win_size - overlap
        if len(time_series) < win_size:
            return {"block_skipped": 0}, pd.DataFrame()
        block_data, last_K = [], None
        block_skipped = 0
        num_windows = (len(time_series) - win_size) // step + 1
        for i in range(num_windows):
            window_data = time_series[i * step: i * step + win_size]
            if not np.isfinite(window_data).all():
                block_skipped += 1
                block_data.append({"window_id": i, "K_metric": np.nan, "deltaK": np.nan, "block_skipped": 1})
                continue
            K_metric = spectral_entropy(window_data)
            deltaK = K_metric - last_K if last_K is not None else 0.0
            block_data.append({"window_id": i, "K_metric": K_metric, "deltaK": deltaK, "block_skipped": 0})
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
