# DOFT — Phase-1 QA Incident Report (Develop Branch)

**Project:** DOFT — Delayed Oscillator Field Theory  
**Branch:** `develop`  
**Report date:** 2025-08-25 (America/Argentina/Buenos_Aires)  
**Reviewer:** Independent QA (protocol conformance & traceability)  

---

## 1) Executive summary

**Status:** ❌ **Not ready — do not proceed to long runs yet**

The latest short run surfaced **instrumentation defects** that invalidate Phase-1 conclusions:

1) **c_eff measurement is unphysical** — all `ceff_pulse` values are **negative**.  
2) **LPC logging missing** — `lpc_deltaK_neg_frac` and `lpc_vcount` fields are **empty**.  
3) **Experiment design incomplete** — only (a, τ) ∈ {(1.0,1.0), (1.0,1.2)} were tested; the 3×3 sweep was **not executed**.

**Bottom line:** this short run is a *useful* debug signal. We must fix measurement & logging before any long computation.

---

## 2) Findings & diagnosis

### F-1. `ceff_pulse < 0` (physically impossible)
- **Symptom:** All measured propagation speeds are negative.  
- **Likely causes (at least one applies):**
  - **Signed distance bug:** using a signed axis coordinate (e.g., `x_peak - x_center`) instead of a **radius**/norm.  
  - **Front detection failure:** thresholds too high or too early decay → the detected “front” time decreases with radius (nonsense) or produces invalid crossings.  
  - **No hysteresis / noisy crossings:** the detector triggers, un-triggers, and re-triggers due to noise, flipping the slope sign.  
- **Impact:** C-1 (emergent-c) cannot be evaluated.

### F-2. LPC metrics not recorded
- **Symptom:** `lpc_deltaK_neg_frac` and `lpc_vcount` missing in `runs.csv`; no window-level entries in `blocks.csv`.  
- **Likely causes:** spectral-entropy loop not executed, detrending/PSD step not reached, or writer not called.  
- **Impact:** C-3 (closed-passive LPC) cannot be evaluated.

### F-3. Incomplete parameter sweep
- **Symptom:** `runs.csv` only shows (a, τ) = (1.0,1.0) and (1.0,1.2).  
- **Impact:** Even with correct measurements, a 2-point set **cannot** validate slope/intercept required by C-1.

---

## 3) Corrective Action Plan (developer-facing)

### MISSION #1 — **Fix the pulse-front measurement** *(priority: absolute)*

**Goal:** `ceff_pulse` must be **positive, stable, and monotone-with-radius** on a basic test.

**Reference algorithm (implementation-agnostic):**
1. **Inject a clean pulse** at `t=0` (Gaussian in space; SNR > 10× ξ).  
2. **Define thresholds** relative to the noise floor: `T ∈ {1σ, 3σ, 5σ}` above ξ (use a baseline window to estimate ξ).  
3. **Per direction (X/Y/Z) and per radius r:**  
   - At each time step, compute **envelope** of the field along that line (e.g., magnitude or Hilbert envelope).  
   - Mark the **first time** `t_cross(r,T)` such that `max_amplitude(line_at_radius_r, t) ≥ T`.  
   - **Enforce causality:** `t_cross(r+Δr,T) ≥ t_cross(r,T)`; if violated, raise the threshold adaptively or use hysteresis (enter at 3σ, exit at 1σ).  
4. Fit a line **radius vs. time** (or time vs. radius) on the valid segment before any boundary reflection:  
   - **Speed**: `ceff_dir_T = Δradius / Δtime`, **must be > 0**.  
   - Aggregate across thresholds/directions → `ceff_pulse` and **95% CI** (bootstrap over seeds × thresholds × directions).  
5. **Stop before edges:** ensure the front is < 25% of the box size away from boundaries (or use absorbing/sponge BCs).

**Sanity micro-tests (must pass locally before long runs):**
- **1D canonical test:** on a known linear wave (or reduced DOFT kernel) with target `c*`, detector must yield `|ceff_pulse − c*|/c* < 1%`.  
- **No negative speed:** assert `ceff_pulse > 0` for all directions/thresholds.  
- **Monotone arrival:** `t_cross(r)` is non-decreasing in `r`.

---

### MISSION #2 — **Implement LPC logging exactly as per the data contract**

**What to compute per sliding window (Welch PSD, Hann, linear detrend):**
- **Spectral entropy:** \( K = -\sum_\omega \tilde S(\omega)\,\log\tilde S(\omega) \) with the PSD normalized to 1.  
- **Delta:** `deltaK = K[n] − K[n−1]`.

**What to record (must write both files):**
- **`blocks.csv` (window-level):**  
  `run_id, window_id, t_start, t_end, K_metric, deltaK, detrend, welch_params`
- **`runs.csv` (run-level summary):**  
  - `lpc_ok_frac` = fraction of windows where `deltaK ≤ 0`.  
  - `lpc_vcount` = **count** of LPC brake activations (should be 0 in closed–passive).  
  - **No NaNs** anywhere; reject run if found.

**Closed–passive mode (Phase-1):** γ ≥ 0, **no drive**, passive kernel. Expect **lpc_vcount = 0** and `lpc_ok_frac ≥ 0.95`.

---

### MISSION #3 — **Run a short but complete 3×3 verification sweep**

**Purpose:** confirm sensitivity to **a** and **τ** and validate LPC logging *before* any long jobs.

**Matrix (≥ 1 seed per point just for verification):**
- **G1 (fixed ratio):** (a, τ) = (1.0,1.0), (1.2,1.2), (1.5,1.5) → expect **constant** `c_eff`.  
- **G2 (↑a, τ fixed):** (1.0,1.0), (1.2,1.0), (1.5,1.0) → expect `c_eff ↑` ∝ a.  
- **G3 (↓τ, a fixed):** (1.0,1.0), (1.0,0.8), (1.0,0.67) → expect `c_eff ↑` ∝ 1/τ.

**What to verify on these 9 runs:**
- `ceff_pulse` **positive** and scales with `a/τ` (trend visible even with 1 seed).  
- `runs.csv` includes isotropy fields (`ceff_iso_x/y/z`) and LPC summaries (`lpc_ok_frac`, `lpc_vcount`).  
- `blocks.csv` is populated for every run.

---

## 4) CI / safeguards to prevent recurrence

Add or update lightweight checks (local or CI):

- **CI-1: ceff positivity:** fail if any `ceff_pulse ≤ 0`.  
- **CI-2: LPC fields present:** fail if `lpc_ok_frac` or `lpc_vcount` missing; fail on NaNs.  
- **CI-3: sweep completeness:** for Phase-1 jobs, assert the 3×3 set exists (9 combos) or an explicit `--dryrun` flag is set.  
- **CI-4: boundary guard:** warn if front detection happens within N cells from boundaries (unless BCs are absorbing+sponge).

---

## 5) Additional stability recommendations (not blocking, but helpful)

- **CFL-like sanity:** ensure the step `dt` and effective `c` obey a stability margin (e.g., `c*dt/a <= 0.5` in linearized tests).  
- **Threshold robustness:** compute ξ (floor) from a pre-pulse window; use hysteresis (enter 3σ, exit 1σ) to avoid flicker.  
- **Determinism:** float64 end-to-end; fixed seeds and thread counts; GPU nondeterministic modes off; record in `run_meta.json`.  
- **Unit “energy-decay” smoke test:** with γ>0, ξ=0 and no drive, assert monotone energy decay (exports a boolean).

---

## 6) Current decision & next gate

**Decision:** ❌ **Hold** — Phase-1 is **not** ready for long/production runs.

**Next approval gate (for QA to flip to ✅ “Ready”):**
1) Short 3×3 sweep re-run with **fixed pulse detector** → `ceff_pulse > 0` and trend vs `a/τ` visible.  
2) **LPC logging present** in both `runs.csv` and `blocks.csv`; in closed–passive: `lpc_vcount = 0`, `lpc_ok_frac ≥ 0.95`.  
3) No NaNs; metadata complete in `run_meta.json`.

Once these three are delivered under `runs/phase1/<date_tag>/...`, QA will issue the **Phase-1 Readiness** approval.

---

## 7) Attachments / required artifacts for the next review

Please include (for the short 3×3 sweep):
- `runs.csv`, `blocks.csv`, `run_meta.json`  
- Plots:  
  - Scatter `c_eff` vs `a/τ` with 95% CI band and linear fit summary.  
  - Isotropy table (`ceff_iso_x/y/z`, `Var(c)/c^2`, `max Δc/c`).  
  - Time series of `K_metric` and histogram of `deltaK` (closed–passive).

---

**Reviewer:** _Independent QA_  
**Date:** 2025-08-25
