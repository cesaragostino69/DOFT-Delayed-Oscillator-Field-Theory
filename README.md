# DOFT — Delayed Oscillator Field Theory
_A research program on emergent spacetime, gravity, and quantum signatures from networks of delayed oscillators_

> “Order is memory made visible. Chaos is the fuel that keeps memory from fading.” — DOFT Axiom A0 (Law of Chaos Preservation)

---

## Table of Contents

- [What this repository is](#what-this-repository-is)
- [Theory snapshot (v1.3)](#theory-snapshot-v13)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
  - [Environment](#environment)
  - [Reproduce a short sweep](#reproduce-a-short-sweep)
  - [Self-averaging report](#self-averaging-report)
  - [Prony fit of the memory kernel](#prony-fit-of-the-memory-kernel)
  - [LPC check](#lpc-check)
- [Data schemas](#data-schemas)
- [Analysis playbook](#analysis-playbook)
  - [Self-averaging](#self-averaging)
  - [Prony kernel + bounds](#prony-kernel--bounds)
  - [Inhomogeneous noise (“grumps are louder?”)](#inhomogeneous-noise-grumps-are-louder)
  - [LPC tests](#lpc-tests)
- [Current takeaways](#current-takeaways)
- [Roadmap](#roadmap)
- [Governance & workflow](#governance--workflow)
- [Reproducibility](#reproducibility)
- [Citing this work](#citing-this-work)
- [License](#license)
- [Code of Conduct](#code-of-conduct)
- [Open questions](#open-questions)
- [A final note](#a-final-note)

---

## What this repository is

This repo hosts the **working specification, analysis scripts, and datasets** for **Delayed Oscillator Field Theory (DOFT)** — a bottom-up framework where **space, time, gravity, and quantum-like spectra** emerge from a **network of coupled oscillators with finite (possibly state-dependent) delays**.

The program is collaborative and adversarial-friendly:

- **Evaluators** (OpenAI & Google): propose tests and falsification criteria  
- **Developer** (Google): implements reference simulators and analysis utilities  
- **Auditor** (OpenAI): reviews code, stats hygiene, reproducibility  
- **Runner**: executes sweeps and aggregates evidence

**Status:** research in progress. Goal: **falsifiable, cross-domain predictions** (atomic spectra, analogue gravity, self-averaging of effective constants) with transparent pipelines and error bars.

---

## Theory snapshot (v1.3)

**A0 — Law of Chaos Preservation (LPC).**  
In closed subsystems the “chaos budget” (functional `𝒦` from positive Lyapunov density / spectral entropy) is conserved; in open subsystems it balances inflow/outflow and dissipation. Order emerges by **channeling** chaos into coherent structures (“clumps”).

**A1 — Local delayed dynamics.**  
Each node \(i\) carries complex amplitude \(q_i(t)\) with natural frequency \(\omega_i\), damping \(\gamma_i\), delayed couplings \(K_{ij}\), phases \(A_{ij}\), and noise floor \(\xi_i(t)\):
\[
\ddot q_i + 2\gamma_i \dot q_i + \omega_i^2 q_i \;=\; \sum_j K_{ij}\,\Re\!\left[e^{iA_{ij}} q_j(t-\tau_{ij})\right] \;+\; \mathcal{N}_i[q] \;+\; \xi_i(t).
\]

**A2 — Space/time as delay-geometry.**  
The matrix \(\{\tau_{ij}\}\) defines causal reach and effective metric; **no background** space is assumed a priori.

**A3 — Holonomies as observables.**  
Only **loop phases** (Wilson-like) are gauge-invariant observables. Local phases are unphysical; loop holonomies encode “fields”.

**A4′ — Emergent quantum floor.**  
A minimal action-per-cycle \(A_0=2\pi\hbar_{\text{eff}}\) emerges from chaotic micro-dynamics and memory; \(\hbar_{\text{eff}}\) is measured as \(\sigma_Q\sigma_P\) in coarse-grained cells.

**A5 — Gravity from state-dependent delays.**  
State dependence \(\tau_{ij}[q,\dot q]\) yields an effective index \(n_{\text{eff}}(x)\propto \tau(x)\). Gradients bend rays and dilate time; horizons appear where flow exceeds group speed \(v_g=a/\tau\).

**A6 — Coarse-graining to PDE with memory.**  
Block-averaging produces a wave-type PDE with **memory kernel** \(M(x,t-t')\) and \(n_{\text{eff}}(x)\). Distributed delay is approximated by a **Prony chain** (sum of exponentials) with explicit error control.

**Headline predictions (falsifiable):**

- **Rydberg/QDT scaling collapse.** Family spectra collapse under \(\nu_m \mapsto \nu_m/(R_{\text{eff}} n_*^3)\); defect slopes correlate with ionic polarizability.  
- **Analogue Hawking from delay gradients.** \(T_H \propto \partial_x\!\left[v_{\text{flow}}-v_g\right]\) with \(v_g=a/\tau(x)\).  
- **Matter = antimatter gravity.** Curvature depends on \(|\tau|\), not on phase sign; same free-fall within error bars.  
- **Self-averaging of \(\hbar_{\text{eff}}\) and \(c_{\text{eff}}\).** Scaling exponents diagnose homogeneous vs clustered noise and set bounds for constant-ness.

---

## Repository layout

~~~text
DOFT/
├─ README.md
├─ MANIFESTO.md
├─ data/
│  ├─ raw/
│  │  ├─ long_merged.csv
│  │  ├─ Raw_selfavg_csv.csv
│  │  ├─ table_full.csv
│  │  ├─ summary.csv
│  │  └─ DOFT_summary__gamma__xi_aggregates_.csv
│  └─ processed/
├─ scripts/
│  ├─ simulate_doft.py
│  ├─ fit_prony_kernel.py
│  ├─ self_averaging.py
│  ├─ hbar_probe.py
│  ├─ lpc_check.py
│  └─ community_noise.py
├─ notebooks/
│  ├─ 01_theory_sanity.ipynb
│  ├─ 02_selfavg_report.ipynb
│  └─ 03_prony_and_bounds.ipynb
├─ configs/
│  ├─ default.yaml
│  └─ sweeps/
│     ├─ sweep_gamma_xi.yaml
│     └─ sweep_memory.yaml
├─ tests/
│  ├─ test_kernel_errors.py
│  └─ test_selfavg_regression.py
├─ CITATION.cff
├─ LICENSE
└─ CODE_OF_CONDUCT.md
~~~

> If your current tree differs, keep filenames consistent with the **Data schemas** below so scripts work out-of-the-box.

---

## Quick start

### Environment

- **Python** ≥ 3.10  
- Core: `numpy`, `scipy`, `pandas`, `numba`, `matplotlib`, `networkx`  
- Optional (speed / autodiff): `jax[cpu]` **or** `pytorch`  
- Community detection (optional): `python-louvain`

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -U numpy scipy pandas numba matplotlib networkx python-louvain
# optional accelerators
# pip install "jax[cpu]"
# OR
# pip install torch
~~~

### Reproduce a short sweep

~~~bash
# Writes CSVs into ./data/raw
python scripts/simulate_doft.py \
  --config configs/sweeps/sweep_gamma_xi.yaml \
  --steps 200000 \
  --sample-every 500 \
  --seed 42
~~~

### Self-averaging report

~~~bash
# Estimate scaling exponent β from block averages d∈{2,4,8,16,32}
python scripts/self_averaging.py \
  --input data/raw/long_merged.csv \
  --metric hbar_eff \
  --divisions 2 4 8 16 32 \
  --group-by gamma xi \
  --out data/processed/selfavg_hbar.csv

python scripts/self_averaging.py \
  --input data/raw/long_merged.csv \
  --metric c_eff \
  --divisions 2 4 8 16 32 \
  --group-by gamma xi \
  --out data/processed/selfavg_c.csv
~~~

### Prony fit of the memory kernel

~~~bash
python scripts/fit_prony_kernel.py \
  --input data/raw/long_merged.csv \
  --trace-field q_t \
  --order 3 5 \
  --out data/processed/prony_fits.parquet \
  --report data/processed/prony_report.md
~~~

### LPC check

~~~bash
python scripts/lpc_check.py \
  --input data/raw/long_merged.csv \
  --metric lyapunov_density \
  --window 2048 \
  --stride 256 \
  --out data/processed/lpc_timeseries.csv
~~~

---

## Data schemas

> If a column isn’t present in your CSVs, scripts simply ignore it. Missing **required** columns will raise clear errors.

### `long_merged.csv` (time-series aggregates per run)

| Column | Type | Meaning |
|---|---:|---|
| `rid` | int | run id / replicate |
| `t` | float | simulation time |
| `gamma`, `xi`, `K` | float | damping, base-noise, mean coupling |
| `a_mean`, `tau_mean` | float | mean link length & delay |
| `c_eff` | float | emergent group speed `a_mean / tau_mean` |
| `sigma_Q`, `sigma_P` | float | coarse-grained std’s |
| `hbar_eff` | float | `sigma_Q * sigma_P` (per block/window) |
| `beta_dyn` | float | dynamic anisotropy indicator |
| `anisotropy` | float | structural anisotropy |
| `lpc_rate` | float | d𝒦/dt estimate (≈0 if closed) |
| `lpc_violations` | int | count of LPC violations (diagnostic) |
| `brake_count` | int | count of “copy-brake” interventions |
| `notes` | str | free-text tags for configs |

### `Raw_selfavg_csv.csv` (block-averaged diagnostics)

| Column | Type | Meaning |
|---|---:|---|
| `rid`, `gamma`, `xi` | mixed | grouping keys |
| `d` | int | block division (2,4,8,16,32) |
| `metric` | str | `hbar_eff` \| `c_eff` \| … |
| `mean_value`, `var_value` | float | per-block stats |
| `rel_var` | float | `var / mean^2` used for scaling fits |

### Summaries: `table_full.csv` / `summary.csv` / `DOFT_summary__gamma__xi_aggregates_.csv`

- `hbar_eff_mean_d2`, `hbar_eff_mean_d4`, `...`  
- `hbar_beta`, `hbar_beta_r2` (from multi-d log–log regression; **avoid 2-point fits**)  
- `c_beta`, `c_beta_r2`  
- `lpc_rate_mean`, `lpc_rate_std`  
- `brake_count`, `lpc_violations`  
- `fig_paths` (optional)

---

## Analysis playbook

### Self-averaging

1. Compute \(\text{rel\_var}(d)\) for \(d\in\{2,4,8,16,32\}\).  
2. Fit: \(\log \text{rel\_var} = -\beta \log N_d + \text{const}\) with \(N_d \propto\) number of blocks.  
3. Interpret \(\beta\): **≈1** strong self-averaging (homogeneous noise); **0<β<1** weak; **≈0** non-self-averaging (critical clustering).  
4. Report CIs (bootstrap over runs). Reject 2-point “perfect” fits (R²=1 is meaningless with two points).

### Prony kernel + bounds

- Fit \(M(t) \approx \sum_{m=1}^M w_m e^{-t/\theta_m}\) (orders \(M=3,5\)).  
- Provide **a priori** truncation bound (Laplace remainder) and **a posteriori** residual MSE on held-out segments.  
- Compare **local** damping from Prony with global \(\gamma\); divergence ⇒ **state-dependent** memory (A5) beyond fixed kernels.

### Inhomogeneous noise (“grumps are louder?”)

- Build graph with weights \(w_{ij} = 1/\tau_{ij}\) or \(K_{ij}\).  
- Detect communities (Louvain).  
- For each community \(C\), compute distribution of \(\hbar_{\text{eff}}(C)\) and correlate with size \(|C|\).  
- Positive slope supports an **inhomogeneous ħ** floor concentrated in coherent clumps.

### LPC tests

- Track chaos functional 𝒦 (Lyapunov density or spectral entropy) in windows.  
  - Closed: \(\dot 𝒦 \to 0\) (within numerical tolerance)  
  - Open: \(\dot 𝒦 \approx \Phi_{\text{in}}-\Phi_{\text{out}}-\mathcal{D}\)  
- Correlate **brake_count** with spikes in 𝒦 to verify the copy-brake is a stabilizing feedback, not masking.

---

## Current takeaways

- **Encouraging trend:** preliminary self-averaging of \(c_{\text{eff}}\) improves with mild damping and modest noise; \(\beta_c \to 1\) over regions of (γ,ξ).  
- **ħ\_eff scaling needs depth:** with only \(d=\{2,4\}\) the exponent is unstable (even negative). Extending to \(d=\{2,4,8,16,32\}\) yields meaningful β with error bars.  
- **LPC consistency:** observed `lpc_rate` near zero when “brake” inactive; excursions align with interventions.  
- **Anisotropy vs noise:** structural anisotropy decreases with ξ; dynamic anisotropy tracks memory timescales; both shape \(n_{\text{eff}}\) maps.

_(Update this section as longer runs land.)_

---

## Roadmap

- **R1 — Robust self-averaging scans:** full \(d\)-ladder fits for \(\beta_{c},\beta_{\hbar}\) with CIs across (γ,ξ,K); publish heatmaps & tables.  
- **R2 — Prony kernel audit:** orders 3/5/7, remainder bounds, local-vs-global damping comparison; decide on state-dependent kernels.  
- **R3 — Community-noise assay:** Louvain + \(\hbar_{\text{eff}}(C)\) vs \(|C|\) regression; quantify “grumps louder”.  
- **R4 — Analogue-gravity check:** controlled delay gradient, measure \(v_g=a/\tau(x)\), verify \(T_H \propto \partial_x[v_{\text{flow}}-v_g]\).  
- **R5 — Atomic spectra cross-checks:** Rydberg/QDT collapses and δ-slope vs polarizability across Na, Rb, Cs; propagate uncertainties.

---

## Governance & workflow

- **Evaluators:** define tests & falsification thresholds  
- **Developer:** implements simulators & analysis  
- **Auditor:** reviews code & stats  
- **Runner:** executes sweeps & publishes artifacts

**Issue labels:** `theory`, `simulator`, `analysis`, `data`, `repro`, `infra`  
**PR checklist:** docstrings, unit tests, seeded randomness, deterministic I/O, CSV schema validation, CHANGELOG entry.

---

## Reproducibility

- **Seeds everywhere.** All scripts accept `--seed`.  
- **Deterministic sampling cadence.** Use `--steps` and `--sample-every`.  
- **Schema contracts.** Scripts fail loud on unknown/missing columns.  
- **Artifacts.** Each run writes a `run.json` (config + git hash) next to CSVs.

---

## Citing this work

Create `CITATION.cff`:

~~~yaml
cff-version: 1.2.0
title: "DOFT: Delayed Oscillator Field Theory"
authors:
  - family-names: "Surname"
    given-names: "Name"
date-released: "2025-08-26"
url: "https://github.com/<org>/DOFT"
license: "MIT"
abstract: "A research program on emergent spacetime, gravity, and quantum signatures from networks of delayed oscillators."
~~~

---

## License

Code: **MIT**  
Data (unless noted): **CC-BY 4.0**

---

## Code of Conduct

We follow Contributor Covenant. Be critical, not personal. Adversarial testing is welcome; adversarial behavior is not.

---

## Open questions

1. **Emergent constants:** under what regimes do \(\beta_c,\beta_{\hbar}\to 1\) with tight CIs? When do they fail (critical clustering)?  
2. **Kernel sufficiency:** do fixed-order Prony kernels capture state-dependent delay, or are adaptive kernels required?  
3. **Inhomogeneous ħ:** do larger communities systematically exhibit higher \(\hbar_{\text{eff}}\)?  
4. **Antimatter gravity parity:** can any parity-breaking term in \(\tau[q]\) generate measurable deviations?  
5. **Lorentz emergence:** quantify Lorentz-violation terms in the coarse-grained PDE and their suppression with scale.

---

## A final note

This repository aims to **earn** credibility by making failure modes visible: every figure should trace to a CSV, every claim to a script, every “constant” to a scaling law. If a prediction breaks under a better test, that’s progress.

Happy falsifying.
