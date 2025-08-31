# Iteration 2 — **Phase 2** Protocol: Dynamic, State‑Dependent Delays in DOFT

**Status:** Draft for implementation · **Owner:** Equipo DOFT · **Target branch:** `feature/phase2-dynamic-delays`

> This document specifies the Phase‑2 protocol for DOFT, extending Phase‑1 to **dynamic, state‑dependent delays** (τ) consistent with Axioms A1–A6 (v1.2). It mirrors the structure of Phase‑1 (objectives → data contract → experiments → exit criteria → deliverables → QA checklist) and must be treated as the single source of truth for this phase.

---

## 1) Objective & Scope

**Primary objective.** Validate that the effective propagation speed and geometric response in DOFT emerge when **link delays τ_ij depend smoothly on local state** (q, p), not on fixed step indices. We must:

- Demonstrate that \(c_{\mathrm{eff}} \approx a/\langle\tau\rangle\) still holds when \(\tau_{ij}(t)=\tau^{(0)}_{ij}+\delta\tau_{ij}[q,\dot q]\) (weak, smooth dependence), and quantify isotropy.
- Show passive **LPC** compliance (no creation of chaos) with dynamic delays.
- Provide an operational path to **retardos variables en el tiempo** without exploding memory or violating passivity.

**Out of scope.** Full Planck‑scale (\(\hbar\)) derivations; non‑abelian holonomies; active gain (γ<0) beyond dedicated QA; long‑range memory other than Prony.

---

## 2) Mathematical Model (State‑Dependent Delays)

### 2.1 Definitions
- Node dynamics (coarse‑grain form):
  \[\ddot q_i + \gamma_i\dot q_i + \Omega_i^2 q_i = \sum_j K_{ij}\, q_j\big(t - \tau_{ij}(t)\big) + \Xi_i(t) + M\text{-terms}\]
- **Dynamic delay:** choose one of the equivalent parametrizations (both must be Lipschitz in the weak regime):
  1. **Direct (weak dependence):** \(\tau_{ij}(t)=\tau^{(0)}_{ij}+\alpha\,\mathcal G(q_i,\dot q_i,q_j,\dot q_j)\)
  2. **State‑augmentation (preferred):**
     \[ \dot z_{ij}=-\lambda\big(z_{ij}-\mathcal F(q_i,\dot q_i,q_j,\dot q_j)\big),\quad \tau_{ij}(t)=\tau^{(0)}_{ij}+\beta\,z_{ij}(t) \]

### 2.2 Regularity & Passivity constraints (hard limits)
- Amplitude bound: \(|\delta\tau_{ij}| \le \epsilon_\tau\,\tau^{(0)}_{ij}\) with \(\epsilon_\tau\in[0.05,0.2]\).
- Slew‑rate bound: \(|\dot\tau_{ij}| \le \eta\,\omega_{\text{loc}}^{-1}\) with \(\eta\in[0.05,0.1]\).
- Prony kernel: weights \(w_m\ge 0\), timescales \(\theta_m>0\) (ensures \(\operatorname{Re}\,\tilde M(\omega)\ge 0\)).
- Damping: closed–passive runs require \(\gamma\ge 0\) and **no external drive**.

> **Rationale:** Lipschitz τ and passive memory guarantee well‑posedness and preclude numerical gain from the delay mechanism.

---

## 3) Numerical Realization (IMEX + Dynamic Fractional Delay)

### 3.1 Integrator
- **IMEX (semi‑implicit):** implicit in stiff/passive terms (−γP, elastic), explicit in coupling via \(q_j(t-\tau_{ij}(t))\) and Prony’s memory force (with exact/implicit update for y_m ODEs).
- **Timestep policy (adimensional):**
  - Start \(dt=0.005\); enforce \(dt \le \min(\tau_{\min}/20, 0.02)\).
  - **Step rejection** if: NaN/inf; passive energy increase beyond tolerance; or **fractional index jump** \(|\Delta d_{ij}|>0.25\). On rejection: halve dt and retry.

### 3.2 Accessing \(q_j(t-\tau_{ij}(t))\) without global histories
- **Ring buffer per node** \(j\): length \(L=\lceil\tau_{\max}/dt\rceil + m\) (margin m=3–5).
- **Fractional‑delay interpolation** per edge (i←j): Lagrange/Farrow all‑pass, order 3–5, reading at index \(d_{ij}(t)=\tau_{ij}(t)/dt\). This preserves amplitude in‑band and avoids spurious gain.
- **Small‑mismatch alternative (optional):** if \(|\delta\tau|\ll\tau^{(0)}\) and \(|\dot\tau|\) is small, use first/second‑order expansion around \(\tau^{(0)}\) (still with fractional read at \(t-\tau^{(0)}\)).

### 3.3 Stability guards (passive mode)
- **Energy‑extended monotonicity:** in closed–passive runs, the extended energy (oscillators + Prony states) must **not increase**.
- **Delay‑slew guard:** track and log events with \(|\Delta d|>0.25\). Rate must stay <1% of steps in accepted runs.

---

## 4) Experiments (Phase‑2)

### Exp‑A — c emergente con τ dinámica
- **Setup:** 2D grid, nearest‑neighbors, \(\tau_{ij}(t)=\tau_0\big[1+\alpha\,(\rho_i+\rho_j)/(2\rho_0)\big]\) with \(\rho_k=\tfrac12(q_k^2+p_k^2)\), weak \(\alpha\in\{0,0.05,0.1,0.2\}\).
- **Protocol:** estado basal estable → **pulso gaussiano** centrado → medir frente en 1σ/3σ/5σ, ángulos x/y/diag.
- **KPI:** regression of \(c_{\mathrm{eff}}\) vs \(a/\langle\tau\rangle\) → **slope = 1.0±0.05**, intercept 95% CI contains 0; **anisotropy** < 2%.

### Exp‑B — Gradiente dinámico (gravedad análoga mínima)
- **Setup:** inducir banda de mayor \(\rho\) ⇒ mayor \(\tau(x)\) (vía \(\alpha>0\) o campo lento \(z_{ij}\)).
- **Protocol:** lanzar pulso desde \(\tau\) baja hacia alta; reconstruir \(c_{\mathrm{eff}}(x)=a/\tau(x)\) y \(n_{\mathrm{eff}}(x)=c/c_{\mathrm{eff}}(x)\).
- **KPI:** curvatura/deflexión del frente consistente con gradiente de \(n_{\mathrm{eff}}(x)\) (error dentro del IC del método de frente/interpolador).

### Exp‑S (sanity, previos a producción)
- **S‑0:** \(\alpha=0\) (τ constante) → reproducir Phase‑1 dentro de ±1%.
- **S‑1:** τ(t)=τ₀(1+ε sin Ωt) global, ε=0.1, Ω≪τ₀⁻¹ → error de amplitud <1%, sin creación de energía; \(|\Delta d|<0.25\).
- **S‑2:** gradiente suave \(\tau(x)\) inducido por \(\rho\) → curvatura del frente consistente con \(n_{\mathrm{eff}}(x)\).

---

## 5) Exit Gate — Success Criteria (Go/No‑Go)

- **(DD‑1) c emergente con τ dinámica:** slope \(c_{\mathrm{eff}}\) vs \(a/\langle\tau\rangle\) = **1.0±0.05**; intercept 95% CI contains 0 (Exp‑A).
- **(DD‑2) Isotropía:** **max Δc/c < 2%** entre ejes (x, y, diag) (Exp‑A).
- **(DD‑3) LPC pasiva:** **lpc_ok_frac ≥ 0.95** y **lpc_vcount = 0** en modo pasivo (Exp‑A/B).
- **(DD‑4) Numérica (τ dinámica):** tasa de rechazo por \(|\Delta d|>0.25\) < 1%; S‑1 error de amplitud <1%.

> Failing any DD‑criteria → **No‑Go**; adjust (α, λ, dt policy, interp order) and rerun.

---

## 6) Data Contract (file schema)

### 6.1 `runs.csv` (one row per run)
Required columns (Phase‑2 additions in **bold**):
- `run_id, seed, a_mean, tau_mean, gamma`  
- `ceff_pulse, ceff_pulse_ic95_lo, ceff_pulse_ic95_hi`  
- `ceff_iso_x, ceff_iso_y, ceff_iso_z, ceff_iso_diag, anisotropy_max_pct, var_c_over_c2`  
- `lpc_ok_frac, lpc_vcount, lorentz_window`  
- **`tau_dynamic_on` (bool), `alpha_delay`, `lambda_z`, `interp_order`, `ring_buffer_len`, `dt_max_delta_d_exceeded_count`, `delta_d_rate`**

Field notes:

- `tau_dynamic_on` toggles updates of link delays during a run.
- `alpha_delay` scales how the local field influences the delay.
- `lambda_z` is the relaxation rate of the auxiliary `z` memory used when smoothing delay adjustments.
- `epsilon_\tau` and `\eta` live in `run_meta.json` and control ring-buffer slack and the maximum normalized delay slew, respectively.
- `delta_d_rate` measures the fraction of integration steps where the computed delay change `\Delta d` exceeds the permitted bound.

### 6.2 `blocks.csv` (windowed LPC)
- `run_id, window_id, K_metric, deltaK, block_skipped`

### 6.3 `run_meta.json`
- seeds, (a, τ) grid, dt rules, **τ model (direct vs z‑aug)**, **bounds (ε_τ, η)**, Prony params (w, θ), front thresholds, code version (Git SHA if available), grid topology, boundary policy.

---

## 7) QA Panel (must be included with deliverables)

- **Dispersion plot:** scatter of \(c_{\mathrm{eff}}\) vs \(a/\langle\tau\rangle\) with linear fit, 95% CI for slope/intercept.  
- **Isotropy table:** `ceff_iso_*` and `anisotropy_max_pct`.  
- **LPC panel:** time series of `K_metric`, histogram of `deltaK`, and summary (`lpc_ok_frac`, `lpc_vcount`).  
- **τ dynamics diagnostics:** histogram of `Δd` per step; `delta_d_rate`; distribution of `ring_buffer_len` utilization.

---

## 8) Operational Notes & Risks

- **Interpolation order:** 3–5 is the sweet spot; higher orders can ring with fast τ(t). Use the slew guard.
- **Boundaries:** prefer absorbent; otherwise terminate front analysis before first reflection.
- **Performance:** O(edges) fractional reads; memory O(nodes·L). Track wall‑time vs. Phase‑1; if needed, reduce angles while preserving isotropy KPI.
- **Config asserts:** enforce \(w_m\ge 0,\theta_m>0,\epsilon_\tau,\eta\) limits at config load.

---

## 9) Deliverables (Phase‑2)

1) `runs.csv`, `blocks.csv`, `run_meta.json` with the **Phase‑2 fields**.  
2) QA PDF/Notebook with the **panel in §7**.  
3) One‑page Executive Summary: direct verdict on **DD‑1..DD‑4** with key figures.

---

## 10) Minimal Acceptance Checklist (what reviewers will verify)

- Schema correctness (files present; Phase‑2 fields populated).  
- Exp‑S (S‑0..S‑2) all **pass**.  
- Regressions and KPIs: **DD‑1..DD‑4** satisfied with clear CIs; no mixing of active (γ<0) into passive datasets.

---

**End of Protocol — Phase 2**

