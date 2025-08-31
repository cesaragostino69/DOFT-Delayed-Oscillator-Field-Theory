# DOFT — Delayed Oscillator Field Theory
_A research program on emergent spacetime, gravity, and quantum signatures from networks of delayed oscillators_

**Status:** Research Alpha | Open Methodology | Cross-Lab Collaboration

> “Order is memory made visible. Chaos is the fuel that keeps memory from fading.” — DOFT Axiom A0 (Law of Chaos Preservation)

---

## 1. Overview

This repository hosts the reference implementation, experiments, and analysis pipeline for DOFT—a bottom-up, network-dynamics framework where the building blocks are identical oscillators coupled through retarded links. In DOFT, **space, time, fields, and gravitation are not fundamental objects**; they emerge from the causal pattern of delays and phases on a large graph.

The project's goals are:
-   **Theory-to-data:** Derive falsifiable predictions (scalings, collapse laws, stability bounds) from DOFT’s axioms.
-   **Data-to-theory:** Test those predictions with numerics and public datasets, and report success/failure with code-audited, reproducible runs.

[This is the DORF Manifesto](./MANIFESTO.md)

[This is the DORF Manifesto uses plain language and analogies](./MANIFESTO_EXPLAINED.md)
*. This text uses plain language and analogies to make DOFT accessible to non-specialists. It does not replace the technical formulation or aim to provoke; any simplification is intentional to aid understanding.*

[This is the DORF Manifesto en lenguaje coloquial y analogías](./MANIFESTO_EXPLICADO.md)
*. Este texto usa lenguaje coloquial y analogías para acercar DOFT a lectores no especialistas. No sustituye la formulación técnica ni busca polemizar; cualquier simplificación es intencional para facilitar la comprensión.*

---

## Table of Contents

- [What this repository is](#what-this-repository-is)
- [Theory snapshot (v1.3)](#theory-snapshot-v13)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
  - [Environment](#environment)
  - [Reproduce a short sweep](#reproduce-a-short-sweep)
  - [Self-averaging report](#self-averaging-report)
- [Data contracts](#data-contracts)
- [Validation suite](#validation-suite)
- [Falsifiable predictions](#falsifiable-predictions)
- [Core concepts](#core-concepts)
  - [Oscillators with delays](#oscillators-with-delays)
  - [Prony memory kernel](#prony-memory-kernel)
  - [Copy–brake law](#copybrake-law)
  - [Emergent $c$ (self-averaging)](#emergent-c-self-averaging)
  - [Inhomogeneous $\hbar_{\text{eff}}$ floor](#inhomogeneous-hbar_textheff-floor)
  - [Law of Chaos Preservation (LPC)](#law-of-chaos-preservation-lpc)
- [Experiments: Phase 1 counterproofs](#experiments-phase-1-counterproofs)
- [Open questions](#open-questions)
- [A final note](#a-final-note)

---

## What this repository is

This repo contains:
- A **CPU-only** reference simulator for networks of **delayed oscillators** with finite-memory kernels.
- Optional **dynamic-delay mode** using per-node ring buffers and fractional interpolation.
- A **validation harness** focused on **falsification-first** checks (self-averaging of $c$, LPC in closed systems, etc.).
- A **reporting pipeline** that emits CSV/Parquet plus plots for independent auditing.

All historic patch bundles and hotfixes have been **consolidated** into this repository. The current code represents the latest state; no external patch application is required.

The goal is not to “prove” DOFT, but to **break it quickly** under clean tests. What survives earns attention.

---

## Theory snapshot (v1.3)

DOFT’s working hypothesis:

1. **Substrate:** the world is approximable as a graph of **oscillators** coupled with **propagation delays** $\tau_{ij}$ and **memory kernels** $K_{ij}(t)$.
2. **Creation vs decay asymmetry:** creation of links/locks is effectively instantaneous (threshold process), while decay is **inertial** (slow), encoded via memory.
3. **Copy–brake:** when a local pattern persists, it **copies** (reinforces) into neighbors; as global congestion rises, a **brake** applies (diminishing effective gain).
4. **Emergent constants:** an effective **speed ceiling** $c$ and a **quantum-like floor** $\hbar_{\text{eff}}$ emerge from network statistics, not axioms.
5. **LPC:** in closed systems the **Law of Chaos Preservation** holds: chaos (as a conserved “fuel”) neither spontaneously increases nor vanishes, it **redistributes**.

These are **claims under test**, not final truths.

---

## Repository layout

```
DOFT/
├── README.md                ← quick guide and project goals
├── LICENSE
├── pyproject.toml
├── requirements.txt         ← Python dependencies
├── src/
│   └── doft/                ← Python package with all source code
│       ├── __init__.py
│       ├── models/
│       ├── simulation/
│       ├── analysis/
│       └── utils/
├── scripts/                 ← CLI or maintenance scripts
├── configs/                 ← JSON/YAML configuration files
├── docs/                    ← extensive documentation, guides, papers
└── .gitignore

```

---

## Quick start

### Environment

- Python 3.11 or 3.12
- NumPy, SciPy, pandas, pyyaml, matplotlib
- No GPU code; **CPU-only** by design
- Install dependencies from the root `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Time step selection

The simulator chooses a safe dimensionless time step automatically to
avoid runaway integrations. For each run the step is clamped via

```
dt_nondim = min(0.02, 0.1, tau_nondim/50, 0.1/(gamma_nondim + |a_nondim| + 1))
```

Any configuration requesting a larger step triggers a warning and the
value above is used instead.

## Development Guidelines

Take time to see this document [docs/protocols/iteration1_phase1.md](docs/protocols/iteration1_phase1.md) for guidelines to participate in this project.


```bash
export PYTHONPATH="$PWD/src"   # or pip install -e .
```

### Run experiments from configs

The helper scripts read parameters from JSON files under `configs/`.

- **Closed–passive Phase 1 sweep**

  ```bash
  bash scripts/run_phase1.sh   # uses configs/config_phase1.json
  ```

- **Chaos LPC test**

  ```bash
  bash scripts/run_quick.sh    # uses configs/config_chaos.json
  ```

- **Smoke test**

  ```bash
  DOFT_CONFIG=configs/smoke_test.json bash scripts/run_quick.sh
  ```
  Runs a tiny 16×16 grid for a handful of steps to verify the pipeline.

Each run writes results to a timestamped directory under `runs/passive/` if `gamma ≥ 0`, or under `runs/active/` if `gamma < 0`.

### Self-averaging report

```bash
python -m reports.self_avg --in out/sanity --out out/sanity/report
```

Emits summary CSV/PNG with estimated $\bar c$ and anisotropy $\Delta c / c$.

### Dynamic delay parameters

The simulator can evolve link delays during a run when `tau_dynamic_on` is enabled. Related settings:

- `tau_dynamic_on`: toggle dynamic delay updates.
- `alpha_delay`: scales how strongly the local field `G` modulates the delay.
- `lambda_z`: relaxation rate for an auxiliary `z` state that smooths `G` before applying `alpha_delay`.
- `epsilon_tau`: fractional slack for the delay ring buffer (0.05–0.2) to accommodate changing $\tau$.
- `eta`: maximum allowed normalized change of $\tau$ per step (slew bound).

When dynamic $\tau$ is active, runs report `delta_d_rate`, the fraction of steps where the delay change $\Delta d$ exceeded the allowed bound, forcing the integrator to clamp the time step. See `configs/config_chaos.json` for a sample configuration enabling this mode.

---

## Data contracts

**runs.csv** (one row per run)
- `run_id, seed, n_nodes, d, density, beta_h, beta_c, kernel_order, dt, n_steps, mean_c_hat, std_c_hat, anisotropy, lpc_vcount, copy_events, energy, entropy, timestamp`

**edges.parquet** (graph snapshot)
- `i, j, tau_ij, K_type, K_params, weight`

Contracts are strict; CI checks schema on PR.

---

## Validation suite

We include tests that must pass before trusting any “result”:

1. **Determinism (seeded):** repeated runs with same seed produce same statistics within tolerance.
2. **Finite outputs:** `_step_imex` on CPU produces finite positions/momenta (no NaNs/Infs).
3. **Self-averaging:** estimate $\bar c$ across blocks $(d=2,4,8,16)$; require slopes consistent with $\beta_h,\beta_c \approx 1$.
4. **Anisotropy metric:** unique definition $\Delta c / c = \frac{|c_x - c_y|}{(c_x + c_y)/2}$, CI reported.
5. **Closed vs open LPC:** closed systems keep chaos functional $\mathcal{K}$ stationary (within numeric tolerance); open systems balance flux.

---

## Falsifiable predictions

### P1 — Self-averaging of $c$

**Claim:** In homogeneous regimes and away from critical clustering, blockwise estimates of $c$ **converge** with scale:

$$
\hat c(d) \to \bar c \quad \text{with} \quad \beta_h \approx 1, \; \beta_c \approx 1.
$$

**Test:** compute $\hat c$ on increasing block sizes $d \in \{2,4,8,16,32\}$; fit log–log slope.  
**Fail signal:** $\beta$ slopes $\ll 1$ or drift of $\bar c$ beyond CI with scale.

### P2 — Inhomogeneous $\hbar_{\text{eff}}$ floor

**Claim:** Coherent communities (graph-theoretic) exhibit a **higher noise floor** reminiscent of an effective $\hbar_{\text{eff}}$.

Operationally:

$$
\hbar_{\text{eff}} \propto \text{residual variance after best-fit deterministic dynamics}.
$$

**Test:** detect communities (Louvain) on the coupling graph; compute distribution of residuals per community; correlate with community size.  
**Fail signal:** no correlation, or inverse trend.

### P3 — LPC in closed vs open

**Claim:** In a closed system, the **chaos functional** $\mathcal{K}$ (Lyapunov density or spectral entropy) is **stationary**:

$$
\dot{\mathcal{K}} \approx 0
$$

In open systems:

$$
\dot{\mathcal{K}} \approx \Phi_{\text{in}} - \Phi_{\text{out}} - \mathcal{D}.
$$

**Test:** run paired experiments; compare $\dot{\mathcal{K}}$ statistics.  
**Fail signal:** systematic drift in closed; mismatch in open flux balance.

---

## Core concepts

### Oscillators with delays

Each node $i$ has state $q_i(t)$ with dynamics:

$$
\ddot q_i + \gamma \dot q_i + \omega_i^2 q_i
= \sum_{j} K_{ij} \, q_j(t - \tau_{ij}) + \eta_i(t) - B(\rho(t)) \dot q_i .
$$

- $K_{ij}$: coupling via memory kernel.
- $\tau_{ij}$: propagation delays (graph distances / media).
- $\eta_i$: driving/noise.
- $B(\rho)$: **brake** increasing with congestion $\rho$.

### Prony memory kernel

We use a finite Prony series:

$$
K_{ij}(t) = \sum_{m=1}^M a_m e^{-t/\theta_m} \cdot \mathbf{1}_{t>0}.
$$

This captures **slow decay** (inertia) without full convolution history (efficient).

### Copy–brake law

Local persistence **copies** into neighbors until global **brake** reduces effective gain:

$$
G_{\text{eff}} = G_0 \cdot \frac{1}{1 + \alpha \rho}.
$$

Intuition: faster growth $\Rightarrow$ more congestion $\Rightarrow$ stronger brake.

### Grid spacing and velocity units

The lattice spacing `dx` sets the physical distance represented by a single grid cell.
When `run()` converts radial indices to physical lengths it multiplies by `dx`, so the
effective speeds (`ceff_pulse`, `ceff_x`, `ceff_y`) are reported in units of `dx` per
unit time. Adjusting `dx` rescales the physical units of the simulation without
altering its dimensionless dynamics.

### Emergent $c$ (self-averaging)

Define blockwise estimator $\hat c(d)$ from propagation front statistics.  
We expect convergence with scale in homogeneous regimes.

### Inhomogeneous $\hbar_{\text{eff}}$ floor

Define residual noise floor after best deterministic fit:

$$
\hbar_{\text{eff}} \sim \text{Var}\big(q - \hat q_{\text{det}}\big).
$$

Expectation: **larger communities** store more residuals (higher $\hbar_{\text{eff}}$).

### Law of Chaos Preservation (LPC)

In closed systems:

$$
\dot{\mathcal{K}} \to 0 \quad (\text{within numeric tolerance})
$$

Open systems balance inflow/outflow and dissipation.

---

## Experiments: Phase 1 counterproofs

### E1 — Multi-block $c$ scaling

- Grid sizes $d \in \{2,4,8,16,32\}$.
- Homogeneous params (fixed $\omega,\gamma,K$).
- Fit $\log d \mapsto \log \hat c(d)$; report slope $\beta_h$ and CI.
- Expect $\beta_h \approx 1$; deviations signal inhomogeneity.

### E2 — Anisotropy unification

- Use single metric $\Delta c / c$ with CI across axes.
- If $\Delta c / c$ stays high at large $d$, heterogeneity persists.

### E3 — Community residuals and $\hbar_{\text{eff}}$

- Build graph with weights $w_{ij} = 1/\tau_{ij}$ or $K_{ij}$.  
- Detect communities (Louvain).  
- For each community $C$, compute distribution of $\hbar_{\text{eff}}(C)$ and correlate with size $|C|$.  
- Positive slope supports an **inhomogeneous ħ** floor concentrated in coherent clumps.

### LPC tests

- Track chaos functional 𝒦 (Lyapunov density or spectral entropy) in windows.  
  - Closed: $\dot 𝒦 \to 0$ (within numerical tolerance)  
  - Open: $\dot 𝒦 \approx \Phi_{\text{in}}-\Phi_{\text{out}}-\mathcal{D}$
- Correlate **lpc_vcount** with spikes in 𝒦 to verify the copy-brake is a stabilizer, not a source.

---

## Open questions

1. **Emergent constants:** under what regimes do $\beta_c,\beta_h \to 1$ with tight CIs? When do they fail (critical clustering)?  
2. **Kernel sufficiency:** do fixed-order Prony kernels capture state-dependent delay, or are adaptive kernels required?  
3. **Inhomogeneous ħ:** do larger communities systematically exhibit higher $\hbar_{\text{eff}}$?  
4. **Antimatter gravity parity:** can any parity-breaking term in $\tau[q]$ generate measurable deviations?  
5. **Lorentz emergence:** quantify Lorentz-violation terms in the coarse-grained PDE and their suppression with scale.

---

## Workflow & Governance

This project follows a strict, multi-party workflow to ensure correctness and reproducibility.

-   **Evaluators (OpenAI/Google):** Propose experiments and acceptance criteria via Pull Requests to the `/configs/` directory.
-   **Developer (Google Track):** Implements features and solvers, including unit tests and performance notes.
-   **Code Auditor (OpenAI Track):** Reviews numerical stability, determinism, and metric integrity. Has the authority to block merges that fail audit.
-   **Runner:** Executes merged experiments and publishes signed artifacts to a results store.

## A final note

This repository aims to **earn** credibility by making failure modes obvious, documented, and repeatable. If a prediction breaks under a better test, that’s progress.

Happy falsifying.

