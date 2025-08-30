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
        kernel_params: dict | None = None,
        max_ram_bytes: int | None = None,
        energy_mode: str = "auto",
        log_steps: bool = False,
        log_path: str | None = None,
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
            )
        self.dt_nondim = safe_dt
        self.min_dt_nondim = 1e-6  # Lower bound to prevent infinite halving loops
        self.dt = self.dt_nondim * self.tau_ref  # Actual dt in "physical" units

        # The physical tau is still needed for delay calculation
        self.tau = tau

        # Precompute delay in time steps to avoid repeated calculation
        self.delay_in_steps = self.tau / self.dt

        # Correctly size the history buffer with the new, smaller dt
        self.history_steps = int(math.ceil(self.tau / self.dt)) + 5  # Added safe margin

        self.Q = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.P = np.zeros((grid_size, grid_size), dtype=np.float64)

        # Memory states for Prony-chain kernels (optional)
        self.kernel_params = kernel_params
        if kernel_params and kernel_params.get("weights") is not None:
            n_modes = len(kernel_params.get("weights", []))
            self.y_states = np.zeros((n_modes, grid_size, grid_size), dtype=np.float64)
        else:
            self.y_states = None

        self.max_ram_bytes = max_ram_bytes or MAX_RAM_BYTES

        bytes_per_slice = grid_size * grid_size * np.dtype(np.float64).itemsize
        memory_used = 2 * bytes_per_slice
        if self.y_states is not None:
            memory_used += self.y_states.shape[0] * bytes_per_slice
        available_bytes = self.max_ram_bytes - memory_used
        self.max_history_steps = max(0, available_bytes // bytes_per_slice)
        if self.history_steps > self.max_history_steps:
            raise MemoryError(
                f"Requested history length exceeds available {self.max_ram_bytes / 1024**3:.0f} GB of RAM",
            )
        self.Q_history = np.zeros(
            (self.history_steps, grid_size, grid_size), dtype=np.float64
        )

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
                self.delay_in_steps = self.tau / self.dt
                required_history = int(math.ceil(self.tau / self.dt))
                if required_history > self.max_history_steps:
                    raise MemoryError(
                        f"History buffer exceeds {self.max_ram_bytes / 1024**3:.0f} GB limit. Aborting simulation."
                    )
                if self.history_steps < required_history:
                    new_history_steps = required_history + 5
                    new_Q_history = np.zeros(
                        (new_history_steps, self.grid_size, self.grid_size),
                        dtype=self.Q_history.dtype,
                    )
                    for i in range(self.history_steps):
                        new_Q_history[(t_idx - i) % new_history_steps] = self.Q_history[
                            (t_idx - i) % self.history_steps
                        ]
                    self.Q_history = new_Q_history
                    self.history_steps = new_history_steps
                break

            self.dt_nondim = new_dt
            self.dt = self.dt_nondim * self.tau_ref
            self.delay_in_steps = self.tau / self.dt

            required_history = int(math.ceil(self.tau / self.dt))
            if required_history > self.max_history_steps:
                raise MemoryError(
                    f"History buffer exceeds {self.max_ram_bytes / 1024**3:.0f} GB limit. Aborting simulation."
                )
            if self.history_steps < required_history:
                new_history_steps = required_history + 5
                new_Q_history = np.zeros(
                    (new_history_steps, self.grid_size, self.grid_size),
                    dtype=self.Q_history.dtype,
                )
                for i in range(self.history_steps):
                    new_Q_history[(t_idx - i) % new_history_steps] = self.Q_history[
                        (t_idx - i) % self.history_steps
                    ]
                self.Q_history = new_Q_history
                self.history_steps = new_history_steps


    def _log_step(self, t_idx: int):
        """Store per-step energy and LPC metrics if logging is enabled."""

        terms = compute_energy_terms(
            self.Q,
            self.P,
            self.a_nondim,
            self.y_states,
            self.kernel_params,
        )
