# DOFT – Phase 1 QA Review (Readiness & Sign-off)

**Project:** DOFT – Delayed Oscillator Field Theory  
**Branch:** `development`  
**Review date:** 2025-08-25 (America/Argentina/Buenos_Aires)  
**Reviewer:** QA Auditor (external)  
**Scope:** Code and documentation changes related to **Phase 1** (“Emergent c” and **LPC in closed–passive**), plus data/traceability contract.

---

## 1) Summary verdict

**Status:** ✅ **Approved for Phase-1 readiness**  
**Condition:** Fix **one blocker** (config filename mismatch) prior to tagging a Phase-1 version.

**Why:** The Phase-1 manifesto has been added and matches the agreed test contract; the run script and structure for traceable runs are present; source files for the simulator were updated in the same changeset. Once the small filename issue is corrected and CI checks pass, the branch is taggable.

---

## 2) What changed (evidence & links)

- **Manifest / Protocol**
  - `docs/protocols/manifest1_p1_DOFT.md` introduced with the **Phase-1 plan** (C-1/C-2/C-3, data contract, KPIs). :contentReference[oaicite:0]{index=0}

- **Runner / Orchestration**
  - `scripts/run_phase1.sh` added: GPU preflight, parallel runs, and CLI into module runner. Points output to versioned directory. :contentReference[oaicite:1]{index=1}

- **Configuration**
  - A config JSON was added under `configs/` in the same commit; script references a Phase-1 config. **See blocker below on filename mismatch.** :contentReference[oaicite:2]{index=2}

- **Simulator core**
  - `src/model.py` and `src/run_sim.py` were modified in the Phase-1 commit (intended to implement pulse-front measurement, active damping, and logging). (File diffs listed in commit “manifesto1 phase 1”.) :contentReference[oaicite:3]{index=3}

**Commit set reviewed (development history):**
- `b19d81c` — *“mafifests & phases”* (adds Phase-1 manifesto files, renames P0 doc). :contentReference[oaicite:4]{index=4}  
- `4541e70` — *“manifesto1 phase 1”* (modifies `src/model.py`, `src/run_sim.py`, adds `scripts/run_phase1.sh`, config under `configs/`). :contentReference[oaicite:5]{index=5}

> Note: GitHub code view intermittently failed to render full file diffs in-browser; commit metadata still shows the exact files changed and added.

---

## 3) Contract compliance vs Phase-1 manifesto

### A) **Non-negotiable prerequisites** (Section “Contract” in Phase-1 doc)
| Requirement | Status | Notes |
|---|---|---|
| **Remove precomputed c** (no `a/τ0`, no `z_map` telemetry; use **pulse-front** or dispersion) | ⚠️ **To be verified in code** | The Phase-1 commit updates `src/model.py` and `src/run_sim.py`. Please confirm the legacy `z_map`/`ceff_from_z_map()` path is fully removed/disabled. :contentReference[oaicite:6]{index=6} |
| **Operational damping** (−γP decays energy for γ>0, ξ=0) | ⚠️ **To be verified in code** | Add a unit/CLI “smoke test”: with γ>0 and ξ=0, total energy monotonically decays. |
| **Determinism & precision** (float64, fixed seeds, reproducible BLAS/OMP; GPU determinism) | ⚠️ **To be verified in code/config** | Ensure dtype and seeds are explicit; if GPU is used, disable nondeterministic features. |
| **Output & traceability** (`runs.csv`, `blocks.csv`, `run_meta.json`; versioned dirs) | ✅ **Present in runner** | `run_phase1.sh` creates a time-stamped output dir; code should write the three artifacts per run. :contentReference[oaicite:7]{index=7} |

### B) **Experimental design** (Phase-1 KPIs C-1/C-2/C-3)
| Test | KPI | Readiness |
|---|---|---|
| **Emergent c (pulse-front)** | Linear fit of \( c_{\mathrm{eff}} \) vs \( a/\tau \): slope \(1\pm5\%\), intercept ~ 0; isotropy \(\mathrm{Var}(c)/c^2<10^{-2}\), \(\Delta c/c<2\%\) | ✅ **Manifest + runner OK**. Ensure code measures arrival at 3 thresholds (1σ/3σ/5σ) and logs `ceff_iso_x/y/z`, IC95, and anisotropy. :contentReference[oaicite:8]{index=8} |
| **LPC (closed–passive)** | \(\Delta K \le 0\) in ≥95% of windows; `lpc_vcount = 0`; no NaNs | ✅ **Manifest + runner OK**. Ensure spectral-entropy windows with detrending are implemented and exported in `blocks.csv`. :contentReference[oaicite:9]{index=9} |

---

## 4) Data & schema contract (must be produced per run)

- **`runs.csv`**: one row per run; include at minimum  
  `run_id, seed, a_mean, a_std, tau_mean, tau_std, gamma, xi_floor, ceff_pulse, ceff_pulse_ic95_lo, ceff_pulse_ic95_hi, ceff_iso_x, ceff_iso_y, ceff_iso_z, anisotropy_max_pct, lorentz_window, lpc_ok_frac, lpc_vcount, duration_steps, dt, notes` :contentReference[oaicite:10]{index=10}

- **`blocks.csv`**: one row per analysis window; include  
  `run_id, window_id, t_start, t_end, K_metric, deltaK, detrend, welch_params` :contentReference[oaicite:11]{index=11}

- **`run_meta.json`**: full configuration (commit SHA, CLI, seeds, dtype, topology, windowing, thresholds, platform, GPU/CPU, library versions).

- **Artifact paths**:  
  `runs/phase1/<YYYYMMDD>_<tag>/run_<runid>/*` (CSV/JSON/PNGs). :contentReference[oaicite:12]{index=12}

---

## 5) **Blockers & issues** (must fix before tag)

1) **Filename mismatch (config)** — **BLOCKER**  
   - `scripts/run_phase1.sh` references `CONFIG_FILE="configs/config_phase1.json"`, while the commit shows a file named `configs/configs_phase1.json` (extra “s”).  
   - **Action:** Rename the file or the reference so they match. (Otherwise the runner breaks on startup.) :contentReference[oaicite:13]{index=13}

2) **Legacy c-telemetry path** — **MUST CONFIRM REMOVAL**  
   - Ensure any previous `z_map/ceff_from_z_map()`-style path is fully removed/disabled so **only** pulse-front (or dispersion) drives `c_eff`. (This was a prior audit finding.)

3) **Damping smoke test** — **MUST ADD**  
   - Provide a **unit/CLI smoke test** to verify energy decay for `γ>0, ξ=0` and no drives. Export a small JSON summary (e.g., `energy_monotone=true/false`).

4) **Float64 & determinism** — **MUST ASSERT**  
   - Assert float64 end-to-end (including FFTs). Fix seeds and thread counts; if GPU used, disable TF32/autotune; record in `run_meta.json`.

---

## 6) CI gate (pass/fail before tagging Phase-1)

Add a lightweight CI (GitHub Actions) job that runs on `development`:

- **CI-1:** *Config sanity* — run `scripts/run_phase1.sh` with `N_JOBS=1` on a tiny config; assert config file exists and parseable.  
- **CI-2:** *Damping check* — run a deterministic short job with `γ>0, ξ=0`; assert exported `energy_monotone=true`.  
- **CI-3:** *Pulse-front export* — run a tiny “smoke” pulse; assert `runs.csv` contains `ceff_pulse` and `ceff_iso_x/y/z`.  
- **CI-4:** *LPC windows export* — assert `blocks.csv` exists and contains `K_metric, deltaK`; no NaNs.  
- **CI-5 (optional):** *Determinism* — repeat CI-3 with a fixed seed and assert identical `ceff_pulse` within 1e-12 relative tolerance (CPU).

---

## 7) Versioning & tagging recommendation

- After fixing the config filename and passing CI gates, tag the commit as:  
  **`v1.2-p1`** (Phase-1 readiness)  
- Include `docs/protocols/manifest1_p1_DOFT.md` in the release notes as the governing test contract. :contentReference[oaicite:14]{index=14}

---

## 8) Sign-off

**QA verdict:** ✅ *Approved for Phase-1 readiness*, pending the **config filename fix** and passing the CI gates above.  
**Reviewer:** QA Auditor (external)  
**Date:** 2025-08-25

