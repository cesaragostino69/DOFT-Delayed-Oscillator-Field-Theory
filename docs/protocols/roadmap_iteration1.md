# DOFT — Counter-test: Consolidation in progress

**Role:** Counter-test scientist (critical, proactive, and oriented to falsifiability).  
**Status:** Batch 1 integrated. Awaiting new batches to extend the consolidation before issuing the first **Pathways Report**.

---

## 1) Summarized conceptual framework (DOFT)
- Ontology: network of **elementary oscillators** \(q_i(t)\) coupled with **finite delays** \(\tau_{ij}\).  
- Baseline dynamics (schematic): local oscillator, damping \(\gamma_i\), natural frequency \(\omega_i\), delayed coupling \(K_{ij} q_j(t-\tau_{ij})\), and weak noise \(\xi_i(t)\).
- Emergence: **space, effective fields, and constants** arise from the delays and their organization.
- Bold proposal: **gravity** as **spatial gradient of delays** (instead of metric curvature directly). Light bending would be reinterpreted as spatial variation of \(\tau\).

## 2) Proposed operational connections
- **Atomic spectra (Rydberg-type):** operational mapping from delay parameters \(\tau\) to quantum numbers \(n\) and quantum defect \(\delta\), reproducing observed transition frequencies (per the other chat).

## 3) Key hypotheses to test
**H1 — Inhomogeneous quantum noise**  
- Idea: the background noise \(\xi_i\) is not homogeneous; it concentrates into emergent resonant **“clumps.”**  
- Observable: **self-averaging** exponent \(\beta_h\) via log–log scaling with block size \(d\).  
  - Expectation: \(\beta_h \approx 1\) → homogeneous noise; \(\beta_h < 1\) → **inhomogeneous**.

**H2 — Remnant noise as “network memory”**  
- FDT link: **noise** and **dissipation** \(\gamma_i\) as two sides of the same mechanism.  
- Prediction: persistent (“remnant”) noise leaves a temporal **imprint** (memory) beyond the declared kernel.

## 4) Initial methodological critique (from the team itself)
- Estimating \(\beta_h\) with **only two points** \(d=2,4\) → trivial linear fit (\(R^2=1\) lacks meaning).  
- Adopt **robust methodology**: more scales \(d=2,4,8,16,32,\ldots\)**,** estimators like **Theil–Sen** and **bootstrap** for CIs.

## 5) Reported results (per mentioned files)
- **Raw_selfavg_csv.csv:** \(\beta_{h,\text{boot, mean}} \approx 0.94\).  
  - Provisional interpretation: **weak self-averaging** → **inhomogeneous noise** (consistent with H1), pending confirmation with more scales and controls.
- **long_merged.csv:** `lpc_ok_frac = 1.0` (the simulator’s “copy brake” did not activate).  
  - Associated qualitative observation: **variability** of \(\beta_h\) across runs with identical parameters → possible **self-organized criticality (SOC)**. *Still an inference: will require standard SOC evidence (e.g., power-law avalanches, heavy tails, etc.).*

## 6) Next lines of work (as proposed)
1. **Scaling validation:** widen \(d\) range (\(2,4,8,16,32,\ldots\)) to estimate \(\beta_h\) with CIs and verify power-law behavior.
2. **“Clump experiment”:** community detection (e.g., **Louvain**) and correlation **clump size ↔ average noise power**. Turn qualitative intuition into a **quantitative test**.
3. **Memory analysis:** Prony/ESPRIT on series \(q_i(t)\) to detect **damped modes/residual frequencies** not explained by the declared memory kernel → evidence of **remnant memory**.

## 7) Shadow areas / missing data to close reproducibility
- **Simulator specification:** network size \(N\), topology (all-to-all, sparse, degree distribution), distributions of \(\tau_{ij}\), \(K_{ij}\), \(\gamma_i\), \(\omega_i\), and **noise** \(\xi_i\) (white, 1/f, Gaussian, etc.).
- **Boundary and initialization conditions** (seeds, initial states).  
- **Exact definition** of \(\hbar_{\text{eff}}\) and the **blocking** protocol to measure \(\beta_h\).  
- **Metric details** and windows (how scaling is computed; handling anisotropies if applicable).  
- **SOC criteria** to use (list of observables and specific tests).

> *Note:* These shadows are **not action requests** yet; they are recorded to resolve them when the user indicates the batch submission is closed.

## 8) Register of expert ADVICE (placeholder)
- **Advice #1:** _(pending entry)_
- **Advice #2:** _(pending entry)_

---

### Logbook
- **Batch 1** incorporated. Awaiting next batches to extend the consolidation before the **first counter-test report** (routes, falsification criteria, null controls, and success/failure metrics).



---

## 9) Batch 2 integrated — DOFT v1.2 (draft “short paper”)
**Title:** Spacetime, spectra, and gravity as emergence from a network of delayed oscillators (formalization v1.2)  
**Current experimental focus:** emergence of c and h-bar.  
**Keys:** state-dependent delays, memory-aware coarse-graining, holonomies (U(1)→SU(N)), Rydberg/QDT, analog gravity.

### 9.1 Operational summary
- Discrete network of oscillators with finite delays (some state-dependent).  
- Contributes: (i) memory-compressed coarse-graining (bounded error), (ii) sketches of emergent c, h-bar, and G from micro-parameters, (iii) holonomies to extend to SU(N), (iv) falsifiable predictions (Rydberg collapses; analog Hawking via delay gradients).

### 9.2 Revised axioms (v1.2)
- A1 Local dynamics with memory (DDE with noise and possible mild state-dependence in K and tau).  
- A2 Emergent space: the delay graph defines cones of influence.  
- A3 Gauge via holonomies: loops W(C) from links U_ij (U(1)/SU(N)).  
- A4' Emergent quantum floor (basal noise + minimal action area).  
- A5 State-dependent delays with regularity (Lipschitz).  
- A6 Memory coarse-graining: wave-type equation with kernel M(t); c_eff = a/tau_0; n_eff^2 ≈ 1 - 2 Phi / c^2 + ...

### 9.3 Mathematical framework
- DDE → ODE with memory (Prony): M_r(t)=sum a_m exp(-lambda_m t); bounded error on [0, omega_c].  
- Well-posedness with tau[q] Lipschitz and bounded couplings.  
- Implementations of tau[q]: (i) 1st-order expansion (tau = tau_0 + delta_tau), (ii) state-augmentation with slow variable z.

### 9.4 Emergent constants (c, h-bar, G)
- c: long modes ka<<1 ⇒ omega ≈ v_g |k|, with v_g = a/tau_0(1 - sigma_tau^2/(2 tau_0^2) + ...). Statistical isotropy ⇒ c := a/tau_0.  
- h-bar: from generalized Langevin + FDT. Basal action area A0 = <∮ p dq> = 2 pi sigma_q sigma_p ⇒ h-bar ≡ sigma_q sigma_p. Goal: show sigma_q sigma_p depends only on (tau_0, gamma, link density).  
- G: sensitivity of tau[rho]. With tau(x)=tau_0 + chi_tau * rho ⇒ n_eff^2 ≈ 1 - 2 Phi / c^2, with Phi = alpha_tau rho and, demanding Poisson, G ~ (c^2/4pi) alpha_tau / ell^2.

### 9.5 Gauge and loops (minimal SU(N))
- U_ij in SU(N). <W(C)> ~ exp(-sigma_tau * Area(C)) (confined regime). sigma_tau: “delay tension”.

### 9.6 Observables and correspondences
- QDT/Rydberg with no free knobs: delta_l(n) = delta_l^(0) + sum_L w_l(L) * Delta tau(L) / T_ref(n), with weights fixed by angular projectors, alpha_0 and r_c.  
- Analog gravity/Hawking: k_B T_H = (h-bar/2pi) * | d/dx (v_flow - v_g) |_(horizon), with v_g=a/tau(x).  
- Matter vs antimatter: leading curvature depends on magnitudes of tau, not on phase sign → equivalence.

### 9.7 Falsifiable predictions (P-series)
- P-1 Rydberg collapse when scaling nu by (n* )^3 R_eff.  
- P-2 Times: T_cl ∝ (n*)^3, T_rev ∝ (n*)^4 with constants consistent across Na/Rb/Cs.  
- P-3 delta_l–alpha_0 slope: monotonic and quantifiable with (alpha_0, r_c).  
- P-4 Analog Hawking: T_H[tau(x)] set by delay gradients (BEC/optics).  
- P-5 Matter/antimatter equivalence in gravity (to leading order).

### 9.8 Preliminary results (qualitative)
- Alkali: partial collapses compatible with (n*)^3, (n*)^4 scales.  
- delta_l vs alpha_0: expected trend; w_l fixed only by (alpha_0, r_c).  
- Analog gravity: v_g=a/tau(x) permits reading T_H from gradients.

### 9.9 Limitations and risks
- Universality of h-bar (basal sigma_q sigma_p) still open.  
- Micro→macro for G without ad-hoc functional maps.  
- Lorentzian emergence (lattice artifacts).  
- Smoothness of tau[q] outside the weak regime.

### 9.10 Informational/computational reading (A0-Info)
- “There are no things, there are messages”: oscillators as ports; the physical = patterns traveling with delays and holonomies.  
- Discrete symbolization possible (asynchronous automaton with delays).  
- Conservations ↔ holonomy invariants; topological defects ≈ “particles.”  
- Thermodynamics of information: Landauer and quantum floor; as T→0 matter does not “fall apart” due to the cost of erasing history.  
- Horizons = channel capacity ~0; analog Hawking = promotion of basal noise by gradient.

### 9.11 Validation contract (mandatory KPIs/IO)
**Emergence of c:** c_eff = lim_{k→0} d omega / d k ≈ a/tau_0.  
- KPI: Var(c_eff)/c_eff^2 < 1e-2.  
- Report: sensitivities dc/da, dc/d tau_0.  

**Memory kernel (Prony):** DDE → ODE with M(t) ≈ sum w_m exp(-t/theta_m), M=2–3.  
- KPI: relative kernel error < 10%.  
- Keep geometry explicit (tau[q]).

**Channel metrics:**  
- Local capacity C(x) ≈ 0.5 log2(1 + SNR_eff(x)).  
- Gradient of C collapses with n_eff(x).  
- At horizons: T_H ∝ d_x (v_flow - v_g) with v_g=a/tau(x).

**Minimum inputs (simulate c):** distributions of a and tau_0 (mean/spread), floor noise xi; Prony kernel {(w_m, theta_m)}, M=2–3; anisotropy via sigma_a/a and sigma_tau/tau_0.  
**Required outputs:** c_eff + 95% CI; Lorentzian window (linear k-range); isotropy (Delta c/c along 3 directions); dc/da and dc/d tau_0; touch nothing but a and tau_0; report kernel error.

### 9.12 Emergents (explicit memory and LPC)
- Non-Markovian memory via Prony (M=3; 1:10:100 scales). Alternative: power-law tail s^{-beta} (0<beta<1).  
- Law of Preservation of Chaos (LPC, axiom A0): accessible disorder does not grow via internal dynamics; it only redistributes/degenerates. Operational measures:  
  - Local/global Lyapunov (shadowing): kappa_i(t)=∫ max(0, lambda_i^+(tau)) d tau; K_lambda = sum w_i kappa_i.  
  - Block spectral entropy: H_s = -∫ S~ log S~ d omega; K_H = sum of H_s.  
- Copy brake (if K rises): reduce weights w_m of fast memories (without touching local energy).  
- Continuity of chaos: d_t K + div J_K = Phi_in - Phi_out - D (zero internal production).  
- Consequences: natural bound for T_H; minimal widths; leading-order matter/antimatter equivalence.

### 9.13 Numerical implementation (hard rules)
- Prony (M=3) with error ≤10%; thetas in 1:10:100 ratio.  
- LPC monitoring: choose K_lambda or K_H; in closed setups, enforce K(t) ≤ K(0)+epsilon; activate copy brake if violated.  
- c and isotropy: Var(c_eff)/c_eff^2 < 1e-2.  
- FDT verified between delayed response and floor noise.

### 9.14 Checklist (predictions/controls for future counter-test)
1) Spectral collapse ≤10% after rescaling.  
2) Time relation: T_rev/T_cl ~ n*.  
3) Analog Hawking: T_H ∝ gradient (v_flow - v_g); bound by K(0).  
4) Minimal widths compatible with A4’ + LPC.  
5) No creation of chaos: without drive, Delta K ≤ 0 (numerical tolerance).

> Note: Consolidated without issuing a verdict yet. The counter-test and advance routes will be generated when you signal there are no more batches.

### 9.15 Register of expert ADVICE (batch 2)
- Advice #3: (placeholder)  
- Advice #4: (placeholder)  



---

## 10) Code audit (preliminary)
Files: run_sim.py, model.py, analyze_results.py, utils.py.

### 10.1 Implemented (aligned with DOFT v1.2)
- Prony-type memory (M=3) in model.DOFTModel via auxiliary ODEs Y and term Mterm.
- Map of c: c_eff precomputed as a/tau0 per cell (ceff_map) and mean ceff_bar.
- hbar_eff: std(q) * std(p) in the final window.
- LPC: check with spectral entropy; if it rises, count violation (lpc_viol_frac).
- Anisotropy: simple estimator Delta c / c from directional means.
- Pipelines: run_sim.py sweeps gamma, xi and seed; analyze_results.py plots hbar_eff, LPC, Delta c / c vs gamma.

### 10.2 Critical gaps vs contract (sec 9.11) and axioms
1) Damping gamma: absent from the equation (missing -gamma*P). Gamma is passed and logged but does not affect dynamics.
2) c_eff is not measured via omega(k) dispersion or propagation front; it is derived only from the a/tau0 map.
3) Self-averaging beta_h: missing multi-scale blocking (>=4 points), Theil–Sen and bootstrap with confidence intervals.
4) FDT: missing PSD of floor noise and response/Kernel M(omega) to test the relation.
5) LPC: only spectral entropy; lacks optional Lyapunov/shadowing and documented automatic copy-brake policy.
6) Isotropy: 2-direction estimator; missing 3 directions and/or angular average in k-space.
7) Sensitivities: partial derivatives dc/da and dc/dtau0 not reported.
8) State-dependent delays: not implemented (only per-cell fixed tau0).
9) Universality of hbar_eff: independence from microdetails not tested robustly.
10) SOC: no avalanche/power-tail extraction.
11) Rydberg/QDT and Hawking: no pipelines yet.

### 10.3 Minimal proposed patches (priority)
P1 — Equation with gamma (mandatory): in step_euler use P += dt * ( -gamma*P - K*Q + Mterm + xi ). Allow gamma per cell or scalar; log the effective value.
P2 — Measurement of emergent c (not from map):
- Dispersion: excite a plane mode k and fit omega(k) for |k|->0; slope = c_eff, with 95% CI.
- Front: point delta and measure arrival time radially; c_eff = dist / t in linear regime.
- Report Lorentzian window (linear k-range) and 3-directional isotropy.
P3 — Robust beta_h: blocks d=2,4,8,16..., Theil–Sen, bootstrap N>=1000; report 95% CI and fraction of CIs below/covering/above 1.0.
P4 — FDT: PSD of floor xi per window/block and response (or M(omega)); test S_xixi(omega) = 2 * Re(M(omega)) * Theta.
P5 — LPC + brake: add Lyapunov metric; if Delta K > 0 without drive, reduce weights of fast memories and log before/after.
P6 — Sensitivities and isotropy: numerical derivatives dc/da, dc/dtau0; angular isotropy in k.
P7 — State-dependent tau[q]: tau = tau0 + alpha*Q + beta*P (weak regime) or slow variable z with tau(z); keep passivity.
P8 — SOC / Rydberg / Hawking: add extractors and null controls.

### 10.4 Risks/artifacts to watch
- c may remain fixed by construction if only a/tau0 is used.
- Without gamma one cannot contrast hbar_eff vs dissipation (core of FDT).
- Spectral entropy: use overlapping windows and detrending to avoid false positives from drift.

Generated audit files: doft_code_audit_prelim.md; doft_code_kpi_pattern_counts.csv; code_kpi_scan_results.json.


---

## 11) Developer guide — Specifications and acceptance criteria (no code)
Purpose: guide implementing teams without prescribing concrete implementation. Defines what to measure, how to report it, and when to deem each point valid or refuted.

### 11.1 P1 — Damping gamma in the dynamics
Requirement: the damping term must affect evolution (not just be logged). It may be scalar or per cell.
Minimum tests:
- T-P1.1: stability with gamma>0 and xi=0: monotonically decreasing energy in linear regime.
- T-P1.2: sweep gamma in [0, gamma_max]: hbar_eff vs gamma should show the expected dependency (to be cross-checked by FDT).
Acceptance: run log per run with effective gamma and timestamp; conservation/decline of energy verified in 3 seeds.

### 11.2 P2 — Measuring emergent c (not from the map)
Dispersion mode:
1) Excite a low-k plane mode in 3 orthogonal directions.
2) Measure omega(k) and fit the linear segment near k=0.
3) Extract c_eff as the slope + 95% CI (bootstrap).
Front mode:
1) Gaussian pulse; 2) arrival time at increasing radii; 3) c_eff = dist/t in early regime.
Mandatory reports: Lorentzian window (linear k-range); Delta c / c along 3 directions; partial sensitivities dc/da and dc/dtau0 via controlled finite differences.
Acceptance: Var(c_eff)/c_eff^2 < 1e-2 (isotropy), and agreement with a/tau0 within the 95% CI.

### 11.3 P3 — Robust self-averaging beta_h
Protocol:
- Blocks d in {2,4,8,16,32} (at least 4 effective points) per seed.
- Theil–Sen estimator in log–log space; bootstrap N>=1000 for 95% CI.
- Null control: synthetic homogeneous noise -> beta_h ~ 1 with CIs covering 1.
Acceptance: percentage of runs with entire CI <1, containing 1, and >1. Inhomogeneity is claimed if at least 70% of CIs lie below 1 and the null control does not falsify it.

### 11.4 P4 — Fluctuation–dissipation test (FDT)
Data required per block/window:
- PSD of floor noise S_xi_xi(omega) with the same windowing as the response.
- Response or kernel in frequency M(omega) and its real part.
Check: S_xi_xi(omega) ~ 2 * Re M(omega) * Theta(omega) in the declared band.
Acceptance: mean relative error <10% in-band; spectral coherence >0.8. If it fails systematically, suspect chaos creation -> activate copy brake.

### 11.5 LPC — metric and copy brake
Primary metric: spectral entropy per block with overlapping windows and detrending. Alternative metric: Lyapunov/shadowing for spot confirmations.
Rule: in no-drive scenarios, Delta K <= 0 (numerical tolerance).
Brake: if Delta K rises, reduce weights of fast memories (without touching oscillator energy); log before/after.
Acceptance: at least 95% of windows with Delta K <= 0 without drive; when the brake acts, Delta K returns to <= 0.

### 11.6 Sensitivities and isotropy
Sensitivities: evaluate dc/da and dc/dtau0 with ±1% perturbations and CIs via bootstrap.
Isotropy: angular average in k-space; report maximum and mean Delta c / c.

### 11.7 State-dependent delays (minimal mode)
Minimal spec: tau = tau0 + alpha*Q + beta*P in the weak regime, or slow variable z with tau(z).
Condition: kernel passivity (no net gain).
Acceptance: stability and FDT within tolerances; sensitivity of c to alpha and beta reported.

### 11.8 SOC — verification
Extraction: detect avalanches (over-threshold events) and measure size and duration distributions; fit power law with at least 1 decade of range.
Acceptance: exponent stable under rebinning and threshold change; null controls do not produce heavy tails.

### 11.9 Data contract (files and columns)
Per run (runs.csv): id, seed, date, N, topology, a_mean, a_std, tau0_mean, tau0_std, gamma_eff, xi_floor, kernel_type, M_params, ceff_map_mean, ceff_map_std, ceff_map_min, ceff_map_max, ceff_measured_x, ceff_measured_y, ceff_measured_z, ceff_measured_ic95, lorentz_window, sens_dc_da, sens_dc_dtau, lpc_ok_frac, beta_h_mean, beta_h_ci_lo, beta_h_ci_hi, fdt_err_rel, notes.
Per block (blocks.csv): run_id, d, window_id, beta_point, beta_ci_lo, beta_ci_hi, K_metric, S_xixi_band, ReM_band, block_skipped.
Metadata (run_meta.json): versions, solver, units, windowing, detrending, seeds.

### 11.10 Minimum QA panel (mandatory plots)
1) omega(k) and tangent line at low k (3 directions) with slope CI.
2) Propagation front (distance vs time) with CI.
3) beta_h: log–log cloud with Theil–Sen line; histogram of CIs vs 1.
4) FDT: S_xi_xi(omega) vs 2*Re M(omega)*Theta (valid band).
5) LPC: time series of K and copy-brake activation points.
6) Isotropy: rose or bar diagram of Delta c / c by direction.

### 11.11 Counter-tests matrix (A/B and nulls)
- A/B-1: frozen vs free Prony kernel (same a and tau0) -> c changes only if a/tau0 changes.
- Null-1: homogeneous noise -> beta_h covers 1.
- Null-2: gamma=0, no drive -> K does not grow; if it grows, bug/LPC.
- Stress-1: anisotropy ±5% in a or tau0 -> Delta c / c responds proportionally.

### 11.12 Ready-to-refute / publish criteria
- P1–P4 approved with reproducible reports (seeds and versions).
- Dataset with runs.csv, blocks.csv, run_meta.json and QA panel.
- Explicit statements: in nulls the effect does not appear; in A/B the effect follows DOFT prediction.


---

## 12) External auditor integration — Consolidation and adjustments (no code)
Purpose: incorporate the other auditor’s inputs, align them with our contract (section 11), and close actionable success/failure criteria. No code is proposed; only specifications and KPIs.

### 12.1 Mission 1 — Measure truly emergent c
Matches: agrees with 11.2 (dispersion/front) and the risk of “c fixed by construction.”
Action:
- Remove/ignore any c estimate derived from input maps (a/tau0, z_map, ceff_from_z_map). Keep them only as telemetry/diagnostics, not to estimate c.
- Mandatory pulse experiment (front): central Gaussian pulse; measure front position at several thresholds (1–5 sigmas above the floor). Compute c_eff = r_front / t in early regime and average at the end.
- Recommended dispersion experiment: excite low-k modes in 3 directions; fit omega(k) near k=0. The slope is c_eff with 95% CI.
Acceptance KPIs:
- “Break the constant”: sweep a_mean and tau_mean; the fit of c_eff vs (a_mean/tau_mean) must give slope ~ 1 (±5%) and **intercept** whose **95% CI** contains **0**.
- Isotropy: Var(c_eff)/c_eff^2 < 1e-2 and max Delta c / c < 2% along 3 axes.
- Declared Lorentzian window (k range with ±5% linearity).
Data contract (add to 11.9): ceff_front, ceff_front_ic95, ceff_slope_vs_a_over_tau, ceff_intercept, ceff_r2, ceff_iso_dirs{x,y,z}.

### 12.2 Mission 2 — Prove LPC and trigger the copy brake
Matches: aligned with 11.5.
Critical nuance: using gamma<0 (gain) violates passivity of axiom A0. It serves as QA test of the brake, not as physical validation of LPC in a closed network.
Recommended action (two modes):
- Closed-passive mode (LPC validation): high-entropy initial condition (high-amplitude Gaussian noise), gamma>=0, **no external drive**. KPI: Delta K <= 0 (95% of windows) and lpc_vcount ~ 0 (the brake should not be needed).
- Open/active mode (brake QA): slightly negative gamma (e.g., −0.01) or sustained edge drive; goal: force the brake. KPIs: lpc_vcount > 0, no NaNs, bounded variables, and after braking Delta K returns to <= 0. Log before/after.
Data contract (add): init_chaos_level, gamma_sign_flag, brake_activation_times, deltaK_pre, deltaK_post, stability_flag.
Note: LPC theory can only be claimed with the closed-passive mode; the active mode certifies the brake implementation.

### 12.3 Mission 3 — Include memory in h-bar computation
Matches: complements 11.4 (FDT) and 10.2-(1).
Action: redefine the effective momentum P as a memory-consistent variable with the kernel (Prony). Operationally: reconstruct P from states y_m and kernel weights so that P captures the integral of M(t − t') over Q-dot in-band.
Controls/KPIs:
- Memory dependence: hbar_eff = sigma_Q * sigma_P must show significant partial derivatives with respect to memory parameters (with signs and orders consistent across seeds) for theta_m and w_m.
- FDT compatibility: the new P should improve coherence between S_xixi(omega) and 2*Re{M(omega)}*Theta(omega) (mean relative error < 10% in-band).
- Markovian null: with “short” memory (collapse to Markov), the dependence of hbar_eff on theta_m should vanish within error.
Data contract (add): hbar_sens_theta{1,2,3}, hbar_sens_w{1,2,3}, fdt_err_rel_afterP, markov_null_pass.

### 12.4 Correspondence map (external auditor ↔ contract 11)
- Mission 1 ↔ 11.2 (emergent c) and 11.6 (isotropy/sensitivities).
- Mission 2 ↔ 11.5 (LPC/brake) distinguishing closed-passive vs open/active.
- Mission 3 ↔ 11.4 (FDT/h-bar) plus new sensitivity metric of hbar_eff to memory parameters.

### 12.5 Integrated checklist for the team
1) Remove “masked constants” (do not use input maps to estimate c).
2) Implement pulse and, if possible, dispersion with 95% CI and Lorentzian window.
3) Run LPC in closed mode (validation) and active mode (brake QA), with complete records.
4) Revisit P with memory and close FDT in-band.
5) Update data contract with the new fields from 12.1–12.3.

With this, both audits are unified into a single plan, with KPIs and unambiguous acceptance criteria to guide developers without needing code.


---

## 13) **Definitive** Action Plan — DOFT Counter-test Phase 1 (single reference document)

**Strategic objective:** subject DOFT to its first rigorous refutation test. Show that **c** emerges from network dynamics and that **LPC** holds in a passive system.

> **Consolidation note:** This plan **replaces** any prior instruction in case of conflict (secs. 10–12). The data/column names listed here are the **official** ones for Phase 1.

### 13.1 The “Contract” — Non-negotiable prerequisites
- **Remove precomputed c:** disable any path computing `c_eff` from `a/τ₀` or maps like `z_map`. The **only** source of `c_eff` will be the **pulse measurement**.
- **Activate damping:** the term `-γ·P` must affect dynamics. Minimum test: with `γ>0` and `ξ=0`, **total energy** must **decay**.
- **Data contract (mandatory logging):**
  - **`runs.csv`** — key columns: `run_id, seed, a_mean, tau_mean, gamma, ceff_pulse, ceff_pulse_ic95, anisotropy_max_pct, lpc_deltaK_neg_frac, lpc_vcount`.
  - **`blocks.csv`** — chaos analysis windows: `run_id, window_id, K_metric, deltaK, block_skipped`.
  - **`run_meta.json`** — metadata: versions, analysis parameters (window size, windowing, detrending, topology, etc.).

### 13.2 “Emergent truth” — Phase 1 experiments
**Experiment A — Trial by fire for c**  
**Goal:** refute that `c_eff` is an artifact.
- **Initialization:** stable basal state + **central Gaussian pulse**.
- **Measurement:** `c_eff` via **wavefront** of the pulse.
- **“Break the constant” sweep** (≥5 seeds per point) — 9-point matrix:
  - **Group 1 (ratio 1.0):** `(a, τ) = (1.0,1.0), (1.2,1.2), (1.5,1.5)`
  - **Group 2 (increase a):** `(1.0,1.0), (1.2,1.0), (1.5,1.0)`
  - **Group 3 (decrease τ):** `(1.0,1.0), (1.0,0.8), (1.0,0.67)`

**Experiment B — LPC verdict**  
**Goal:** validate that **LPC** is an intrinsic law in **passive** mode.
- **Initialization:** **Gaussian noise** of high amplitude → high **K(0)**.
- **Dynamics:** **closed and passive** system (`γ ≥ 0`, **no external drive**).
- **Measurement:** **spectral entropy** `K_metric` in overlapping windows with detrending; `deltaK` between consecutive windows.

### 13.3 Exit gate — Success criteria (Go/No-Go)
- **(C-1) c is emergent:** the `c_eff` vs `a_mean/τ_mean` plot (Exp. A) shows a linear fit with **slope = 1.0 ± 0.05** and **intercept** whose **95% CI** contains **0**.
- **(C-2) Isotropic space:** **maximum relative anisotropy** `Δc/c` across axes `< 2%`.
- **(C-3) Passive LPC:** in Exp. B, **fraction of windows** with `deltaK ≤ 0` (`lpc_deltaK_neg_frac`) **≥ 0.95**, and **`lpc_vcount = 0`**.

### 13.4 Mandatory deliverables (Phase 1)
1) **Raw data:** `runs.csv`, `blocks.csv`, `run_meta.json` (with the columns/metadata above).  
2) **QA panel:**
   - Scatter `c_eff` vs `a/τ` with fit and **CI** of slope/intercept.
   - Clear **isotropy** table (`anisotropy_max_pct`).
   - Time series of **K_metric** and **histogram of `deltaK`** (Exp. B).
3) **Executive summary (1 page):** direct verdict on **C-1**, **C-2**, and **C-3**.

#### 13.5 Compatibility (aliases with respect to previous versions of this document)
- `ceff_pulse` ≈ *previously* `ceff_front`; `anisotropy_max_pct` ≈ *previously* `Δc/c (max)`; `lpc_deltaK_neg_frac` ≈ *previously* `fraction ΔK≤0`.  
- Where name discrepancies exist, the nomenclature of §13 **prevails**.
