# Definitive Action Plan — Phase 1 (DOFT Counter-Test)

## 0) Objective and Exit Gate

**Objective:** Demonstrate that **c** is **emergent** (not imposed) and validate **LPC** (Law of Preservation of Chaos) in a **closed–passive** network.

**Phase 1 Go/No-Go (all must pass):**
**(C-1)** Fit of \$(c_{\mathrm{eff}}\) vs. \(a_{\mathrm{mean}}/\tau_{\mathrm{mean}}\)$: **slope = 1 ± 5%** and **intercept ~ 0** (95% CI).
**(C-2)** Isotropy: \$(\mathrm{Var}(c)/c^2 < 10^{-2}\) and \(\Delta c/c\)$ (max across axes) **< 2%**.
**(C-3)** LPC in closed–passive mode: **\(\Delta K \le 0\)** in **≥95%** of windows, **without** triggering the brake (lpc_vcount = 0).

---

## 1) “Contract” (non-negotiable prereqs)

1) **Remove precomputed c**
It is **forbidden** to derive \(c_{\mathrm{eff}}\) from \(a/\tau_0\), **z_map**, or any telemetry. The **only** valid measurement of \(c\) comes from the **front of a pulse** (or, optionally, \(\omega(k)\) dispersion with a Lorentz window).

2) **Operational damping**
The dissipative term \(-\gamma P\) **must** affect dynamics. Minimal test: with \(\gamma>0\), \(\xi=0\), the total energy **must decay**.

3) **Determinism & precision**
- Use `float64` end-to-end.
- Fixed seeds: `seed_global`, `seed_init`, `seed_noise`.
- Reproducible BLAS/OMP (fixed threads), no *fast-math*.
- If GPU: disable TF32/cuDNN autotune for reproducibility.

4) **Output format & traceability (required)**  
- `runs.csv` (one row per run).  
- `blocks.csv` (one row per chaos-analysis window).  
- `run_meta.json` (full run metadata).  
- Artifact naming convention:
  `runs/passive/phase1_run_<timestamp>/*.csv|.json|.png` (use `runs/active/` if `gamma < 0`)

---

## 2) Experimental design (Phase 1)

### A) **Emergent c** — **pulse-front** experiment

- **Init:** stable baseline; centered **Gaussian pulse** (amplitude s.t. SNR > 10× noise floor \(\xi\)).
- **Measurement:** detect **arrival time** of the front at increasing radii using **three thresholds** relative to the floor (\(1\sigma\), \(3\sigma\), \(5\sigma\)).
- **Isotropy:** measure \(c_{\mathrm{eff}}\) along **X, Y, Z** (and a **diagonal** if possible); report \(\Delta c/c\).
- **Lorentz window** (optional, recommended): \(\omega(k)\) via space–time FFT; linear regime with relative curvature < 5%.

**“Break the constant” sweep (3×3, ≥5 seeds per point):**
- **G1 (fixed ratio):** \((a,\tau)= (1.0,1.0),(1.2,1.2),(1.5,1.5)\) ⇒ **expectation:** \(c\) ~ constant.
- **G2 (↑a, fixed \(\tau\)):** \((1.0,1.0),(1.2,1.0),(1.5,1.0)\) ⇒ **expectation:** \(c\uparrow\) proportionally.
- **G3 (↓\(\tau\), fixed \(a\)):** \((1.0,1.0),(1.0,0.8),(1.0,0.67)\) ⇒ **expectation:** \(c\uparrow\) proportionally.

**KPIs A:** linear fit \(c_{\mathrm{eff}}\) vs \(a/\tau\) (slope, intercept, R², 95% CI); \(\mathrm{Var}(c)/c^2\), \(\Delta c/c\), and \(\omega(k)\) linear range if applicable.

---

### B) **LPC** in **closed–passive**

- **Init:** **Gaussian noise** of **high amplitude** (high \(K(0)\)).
- **Dynamics:** \(\gamma \ge 0\), **no drives**, no active boundaries; memory kernel **passive**.
- **Metric:** **spectral entropy** per block (**Welch** or windowed periodogram with **detrending**); overlapping windows (e.g., 50%); \(\Delta K\) between consecutive windows.

**KPIs B:**
- **Closed–passive:** \(\Delta K \le 0\) in **≥95%** of windows; `lpc_vcount = 0`; **no NaNs**.
- **Open/active (optional QA):** with \(\gamma \approx -0.01\) or boundary drive, the **brake** should engage (lpc_vcount>0) and return \(\Delta K\) to \(\le 0\). *Does not count* toward (C-3); this is brake QA.

---

## 3) Data contract (schemas)

### `runs.csv` (suggested types/units)
| field | type | description |
|---|---|---|
| run_id | str | unique id (e.g., `P1_G2_a1.2_tau1.0_s003`) |
| seed | int | main seed |
| a_mean, a_std | float | mean & std of \(a\) [u.l.] |
| tau_mean, tau_std | float | mean & std of \(\tau\) [u.t.] |
| gamma | float | effective damping |
| xi_floor | float | baseline noise (rms) |
| ceff_pulse | float | \(c_{\mathrm{eff}}\) from front (avg over thresholds/directions) |
| ceff_pulse_ic95_lo/hi | float | 95% CI of \(c_{\mathrm{eff}}\) |
| ceff_iso_x/y/z | float | directional \(c\) |
| ceff_iso_diag | float | optional |
| anisotropy_max_pct | float | \(\Delta c/c\) × 100 |
| lorentz_window | str | “k∈[k1,k2], curv<5%” or “NA” |
| lpc_ok_frac | float | fraction of windows with \(\Delta K \le 0\) |
| lpc_vcount | int | count of violations (brake activations) |
| duration_steps | int | total steps |
| dt | float | time step |
| notes | str | notes (e.g., “sponge=on; bc=absorbing”) |

### `blocks.csv`
| field | type | description |
|---|---|---|
| run_id | str | FK to `runs.csv` |
| window_id | int | window index |
| t_start, t_end | float | window time |
| K_metric | float | block spectral entropy |
| deltaK | float | \(K_{n}-K_{n-1}\) |
| block_skipped | int | 1 if window skipped due to invalid data |
| detrend | str | detrending method |
| welch_params | str | summary (nperseg, noverlap, window) |

### `run_meta.json` (template)
```json
{
  "manifest": "Phase1-DOFT-v1.2",
  "code_version": "commit:<hash>",
  "branch": "develop",
  "env": {
    "python": "3.10",
    "numpy": "pinned",
    "scipy": "pinned",
    "pandas": "pinned",
    "torch": "pinned-or-na",
    "device": "cpu|cuda",
    "dtype": "float64"
  },
  "topology": {"grid": [128,128,128], "bc": "absorbing", "sponge_width": 8},
  "kernel_prony": {"M": 3, "thetas": [t1,t2,t3], "weights": [w1,w2,w3]},
  "seeds": {"global": 123, "init": 456, "noise": 789},
  "pulse_amplitude": 0.1,
  "detection_thresholds": ["1sigma", "3sigma", "5sigma"],
  "fft_params": {"st_grid": [64, 256], "k_fit_tol_pct": 5},
  "windowing": {"len_steps": 2048, "overlap": 0.5, "detrend": "linear"}
}

---

## Development Guidelines — iteration1_phase1

**Scope:** Basic rules for contributing to the DOFT project.

1. **PEP 8 compliance:** Ensure all code follows the PEP 8 style guidelines.
2. **Tests required:** Include tests for new features and bug fixes.
3. **Pre-commit checks:** Run `pytest` and other linters before submitting a commit.
4. **Commit hygiene:** Use clear commit messages that describe the change and its motivation.
5. **Design notes:** Document relevant design decisions in code comments or in the documentation.
