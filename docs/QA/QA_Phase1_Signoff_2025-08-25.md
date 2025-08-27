# DOFT — Phase-1 QA Sign-off (Develop Branch)

**Project:** DOFT — Delayed Oscillator Field Theory  
**Branch under review:** `develop`  
**Report date:** 2025-08-25 (America/Argentina/Buenos_Aires)  
**Reviewer:** Independent QA (protocol conformance & traceability)

---

## 1) Decision

**Status:** ✅ **Approved to Proceed to Phase-1 Testing (APT)**

This approval authorizes execution of the Phase-1 experiment matrix and collection of artifacts as defined below. Final release/tagging to `main` remains gated on quantitative pass of **C-1 / C-2 / C-3** and CI checks (Section 5).

---

## 2) Scope & Protocol (what Phase-1 must prove)

**Objective:** show that **c** is **emergent** (not imposed) and that **LPC** holds in a **closed–passive** network.

### Acceptance criteria (must be met by data)

- **(C-1) Emergent c:** Linear fit of \( c_{\mathrm{eff}} \) vs \( a_{\mathrm{mean}}/\tau_{\mathrm{mean}} \) has **slope = 1 ± 5%** and **intercept ≈ 0** (95% CI).
- **(C-2) Isotropy:** \(\mathrm{Var}(c)/c^2 < 10^{-2}\) and **max** \(\Delta c/c < 2\%\) across X/Y/Z (or X/Y/diag).
- **(C-3) Closed–passive LPC:** With \( \gamma \ge 0 \) and no drive, **\(\Delta K \le 0\)** in **≥95%** of sliding windows; **no brake activation** (`lpc_vcount = 0`); no NaNs.

---

## 3) Experiment design (how to measure)

### A) **Emergent c** — pulse-front experiment

- **Init:** stable baseline; centered **Gaussian pulse** (SNR > 10× floor \(\xi\)).
- **Front detection:** first-arrival time at increasing radii using **three thresholds** (1σ / 3σ / 5σ above \(\xi\)); temporal interpolation.
- **Isotropy:** report \(c_x, c_y, c_z\) (and diagonal if available); compute \(\Delta c/c\).
- **Optional:** dispersion \(\omega(k)\) to declare Lorentz window (curvature < 5%).

**3×3 sweep (“break the constant”), ≥5 seeds/point:**
- **G1 (fixed ratio):** \((a,\tau) = (1.0,1.0),(1.2,1.2),(1.5,1.5)\) ⇒ expect **constant** \(c\).
- **G2 (↑a, fixed \(\tau\))**: \((1.0,1.0),(1.2,1.0),(1.5,1.0)\) ⇒ expect \(c \uparrow\) ∝ \(a\).
- **G3 (↓\(\tau\), fixed \(a\))**: \((1.0,1.0),(1.0,0.8),(1.0,0.67)\) ⇒ expect \(c \uparrow\) ∝ \(1/\tau\).

### B) **LPC (closed–passive)**

- **Init:** high-amplitude **Gaussian noise** (large \(K(0)\)).
- **Dynamics:** \( \gamma \ge 0 \), no drives; **passive** memory kernel.
- **Metric:** **spectral entropy** per sliding window (Welch, Hann, linear detrend); compute \(\Delta K\) between consecutive windows.

---

## 4) Data contract (must-produce artifacts)

Write artifacts under a versioned folder:
