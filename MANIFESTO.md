# DOFT Manifesto v1.3

> **Delay‑Oscillator Field Theory (DOFT):** a bottom‑up framework where spacetime, fields, and “constants” emerge from a large network of coupled oscillators with finite, state‑dependent delays. This document consolidates the conceptual core, working equations, testable predictions, critiques addressed so far, and a practical roadmap for simulation and falsification.

---

## 0) One‑page overview

- **Ontology:** The universe is modeled as a graph of identical **oscillators** (nodes) connected by **links** that transmit influence with **finite delays**. No pre‑given spacetime. Causality and geometry **emerge** from the delay structure.
- **Dynamics:** Each node obeys a **delay differential equation (DDE)** with damping, weak noise (“quantum floor”), and nonlinear, delayed coupling to neighbors. Delays can **depend on the state**, enabling backreaction and gravity‑like effects.
- **Emergence:** Coarse‑graining produces an effective **wave equation with memory**, an **index of refraction** $n_\mathrm{eff}(x)$ from delay gradients, and **holonomies** on cycles that play the role of gauge phases.
- **Constants:** The **speed limit** emerges as $c \approx a/\tau_0$ (link scale over minimal delay). An effective **Planck’s constant** $\hbar_\mathrm{eff}$ arises from the invariant action area of steady‑state cycles sustained by the noise floor. The **Newton constant** $G$ maps to the sensitivity of delays to local energy density.
- **Phenomenology:** DOFT reproduces Rydberg‑like spectra via **quantum‑defect analogues** from short holonomy loops, links naturally to **analogue gravity** (Hawking radiation from a horizon in $n_\mathrm{eff}$), and predicts **matter/antimatter** to curve the same geometry.
- **Testability:** We define falsifiable **cross‑domain** tests (atomic spectra ↔ analogue gravity), **self‑averaging** diagnostics, Prony‑based kernel identification, and scaling laws for $\hbar_\mathrm{eff}$.
- **New axiom (A0):** **Law of Preservation of Chaos (LPC):** in closed subsystems the “chaos functional” cannot increase without bound; open subsystems can export/import chaos via fluxes. Order appears as dissipative organization of an initial chaotic budget.

---

## 1) Axioms

- **A0 – Law of Preservation of Chaos (LPC).**  
  There exists a non‑negative functional $\mathcal{K}[q]$ (e.g., Kolmogorov‑Sinai entropy rate proxy or sum of positive Lyapunov exponents) with balance
  
                $$\frac{d\mathcal{K}}{dt} = \Phi_\mathrm{in} - \Phi_\mathrm{out} - \mathcal{D},$$

  where $\mathcal{D}\ge 0$ captures dissipation/regularization. In closed subsystems $\Phi_\mathrm{in}=\Phi_\mathrm{out}=0\Rightarrow d\mathcal{K}/dt\le 0$.

- **A1 – Local delayed dynamics.**  
  Each node $i$ evolves as
  \[
  \ddot q_i + 2\gamma_i \dot q_i + \omega_i^2 q_i + \alpha_i |q_i|^2 q_i
  = \sum_{j\in \mathcal{N}(i)} K_{ij}\, \sin\!\big(\theta_{ij}(t)\big)\, q_j\!\big(t-\tau_{ij}(t)\big) + \xi_i(t).
  \]
  Here $\xi_i$ is a weak, broadband noise (“quantum floor”).

- **A2 – Spacetime as delay‑graph.**  
  The matrix of delays $\{\tau_{ij}\}$ defines cones of influence; geometry and clock rates emerge from path sums of delays.

- **A3 – Holonomy primacy.**  
  Physical invariants are **loop phases**
$$
  W(\ell)=\exp\!\big(i\sum_{(i\!j)\in \ell} A_{ij}\big),
$$
  with link phases $A_{ij}$ accumulated by transport. Local gauge choices are unphysical; only holonomies matter.

- **A4 – Quantum floor.**  
  A minimal stochastic excitation $\xi$ exists. In v1.2+, this noise is treated as **emergent** from high‑dimensional chaotic microdynamics consistent with A0 (Addendum A4′).

- **A5 – State‑dependent delays (backreaction).**  
  $\tau_{ij}=\tau_{ij}\!\big[q,\dot q,\rho\big]$ vary with local fields and coarse energy density $\rho$. Delay gradients generate an effective refractive index and geodesic bending.

- **A6 – Coarse‑graining to continuum.**  
  Block‑averaging over mesoscopic patches yields a **wave‑with‑memory** PDE with kernels $M$ and a spatially varying $n_\mathrm{eff}(x)$.

---

## 2) Working equations and continuum limit

### 2.1 Discrete graph (micro)
\[
\ddot q_i + 2\gamma \dot q_i + \omega_0^2 q_i + \alpha |q_i|^2 q_i
= \sum_{j} K_{ij}\, \sin(A_{ij})\, q_j(t-\tau_{ij}) + \xi_i(t),
\quad \tau_{ij}=\tau_0 + \delta\tau_{ij}[q,\rho].
\]

### 2.2 Memory‑compressed surrogate
Replace pure delays by a finite Prony chain
\[
\dot y_m = -\theta_m y_m + \beta_m q,\qquad
\text{use}\;\; q(t-\tau)\approx \sum_m w_m y_m(t),
\]
with $\sum_m w_m \approx 1$. Identify $\{\theta_m,w_m\}$ from data via **generalized Prony/Vector‑Fitting** to preserve spectra and decay.

### 2.3 Continuum coarse‑graining
On scales $\gg a$ (mean link length):
\[
\partial_t^2 \phi + 2\Gamma \partial_t \phi + \Omega^2 \phi
- \nabla\!\cdot\!\big(c^2 n_\mathrm{eff}^{-2}(x)\nabla \phi\big)
+ \int_0^t M(x,t-t')\,\phi(t')\,dt'
= \Xi(x,t).
\]
- $c \approx a/\tau_0$ is the emergent causal speed.  
- $n_\mathrm{eff}(x)$ encodes averaged delay gradients; horizons occur when **flow – group speed** changes sign.  
- $M$ is the memory kernel inherited from $\{\theta_m,w_m\}$.

---

## 3) Emergent “constants”

- **Speed limit $c$.**  
  $\displaystyle c = \frac{a}{\tau_0}$ up to renormalizations from connectivity and weak disorder. Axiomatically independent of background geometry.

- **Planck constant $\hbar$ as action floor.**  
  The stationary stochastic dynamics with A0+A4′ sustains **limit‑cycle ensembles** whose phase‑space area per cycle stabilizes to
$$
  A_0 \equiv \oint p\,dq \approx 2\pi \hbar_\mathrm{eff}.
$$
  We estimate $\hbar_\mathrm{eff}$ from local fluctuations
  $\hbar_\mathrm{eff}\sim \sigma_Q \sigma_P$
  and test **self‑averaging** scaling
$$
  \mathrm{Var}(\hbar_\mathrm{eff})/\mathbb{E}[\hbar_\mathrm{eff}]^2 \sim N^{-\beta_\hbar}.
$$

- **Newton constant $G$ as delay‑sensitivity map.**  
  In the continuum,
$$
  n_\mathrm{eff}^2(x) \simeq 1 + \alpha_\tau\, \rho(x)
$$
  with $\alpha_\tau = \partial n_\mathrm{eff}^2/\partial \rho$. Linearizing ray bending and matching to weak‑field GR gives a constitutive link $G \propto c^2\,\alpha_\tau$ after proper normalization. This is a working **hypothesis to be derived** from the microdynamics.

---

## 4) Gauge/holonomy and atomic phenomenology

- **Holonomy as short‑loop physics.**  
  Small cycles around “cores” induce phase defects that shift effective Rydberg energies:
  \[
  E_{n\ell} \approx -\frac{R_\mathrm{eff}}{(n-\delta_\ell)^2},\qquad
  \delta_\ell \;\text{from short-loop holonomies and local polarizability}.
  \]
- **Predictions:**  
  (i) Collapse of line families when scaling $\nu_{m}$ by $(n^\*)^3 R_\mathrm{eff}$;  
  (ii) linear trends of $\delta_\ell$ vs. $1/(n^\*)^2$ with slopes correlated to ionic core **static polarizability** $\alpha_0$;  
  (iii) Rydberg wave‑packet times $T_\mathrm{cl}\propto (n^\*)^3$, $T_\mathrm{rev}\propto (n^\*)^4$.

---

## 5) Gravity analogue, horizons, and antimatter

- **Index gradient and geodesics.**  
  Rays follow Fermat in $n_\mathrm{eff}(x)$. A **horizon** forms where a drift $u$ exceeds local wave group speed $v_g=c/n_\mathrm{eff}$.

- **Hawking analogue.**  
  $T_H \propto \big|\partial_x(u-v_g)\big|_{x_H}$. In DOFT this depends on $\partial_x n_\mathrm{eff}$ which is set by delay gradients.

- **Antimatter.**  
  Curvature depends on the **magnitude** of delay gradients, not the sign of phase. Prediction: **same gravitational response** for matter/antimatter at leading order.

---

## 6) Law of Preservation of Chaos (LPC) in practice

We monitor a chaos proxy $\mathcal{K}$ (finite‑window Lyapunov sum, entropy rate, or broadband spectral entropy) and enforce **no blow‑up** in closed subsystems. In open subsystems we allow fluxes but require boundedness via mild **chaos braking** (adaptive reduction of fast‑memory weights) **only** when $\mathcal{K}$ violates target envelopes—this is a *physical surrogate* for dissipation, not a numerical hack.

---

## 7) Critiques addressed (and what remains)

1. **“Analogies ≠ causation.”**  
   We use analogues to choose **falsifiable observables**; we do **not** infer identity. The programme is *predict → test → reject/retain parameters* across domains.

2. **Coarse‑graining rigor.**  
   The PDE with memory is not assumed; we **derive** kernels by Prony/Vector‑Fitting from micro time‑series, then validate via **a posteriori** error bounds (residual energy, BIBO stability, passivity).

3. **DDE intractability.**  
   We replace pure delays by **few‑pole** memory that preserves dominant poles/zeros; convergence is checked by **model‑order increase until residuals plateau**.

4. **Non‑self‑averaging at criticality.**  
   We explicitly **measure** scaling exponents $\beta_\hbar,\beta_c$. If they do not approach 1 (strong self‑averaging) in the regimes claimed to be universal, DOFT **fails** there.

5. **Emergent constants still “sketched.”**  
   Correct: $\hbar, G$ derivations must be **closed**. We outline a stochastic‑invariant action proof for $\hbar_\mathrm{eff}$ and a constitutive derivation for $G$; both are priority items in the roadmap.

---

## 8) Testable predictions & falsification

### 8.1 Cross‑domain tie‑down
- Fit $R_\mathrm{eff}$ and $\delta_\ell(n)$ on **one** alkali (e.g., Cs).  
- **Without retuning**, predict: (i) wave‑packet times; (ii) $\nu$ collapse; (iii) analogue‑gravity $T_H$ vs. $\partial_x n_\mathrm{eff}$ in a **mapped** optical/BEC device.  
- **Fail** of (iii) given success of (i–ii) → refutation of common mechanism.

### 8.2 Self‑averaging of $\hbar_\mathrm{eff}$
- Partition the network into $d=2,4,8,16,32,\dots$ blocks; compute
  $\mathrm{R}(d)=\mathrm{Var}(\hbar_\mathrm{eff})/\mathbb{E}[\hbar_\mathrm{eff}]^2$.
- Fit $\log R = -\beta_\hbar \log N + b$; **expect** $\beta_\hbar \approx 1$ in homogeneous regimes.  
- **Fail** (persistent $\beta_\hbar \ll 1$) → noise is not homogeneous or the coarse‑grain model is wrong.

### 8.3 Hawking analogue slope
- Measure $T_H$ vs. $\partial_x(u-v_g)$; DOFT predicts proportionality with coefficient set by memory‑kernel curvature. Mismatch beyond bounds → reject kernel identification or A5 parametrization.

### 8.4 Antimatter gravity
- Any leading‑order deviation between matter/antimatter geodesics conflicts with DOFT’s phase‑insensitive curvature.

---

## 9) Numerical methodology (for collaborators)

- **Integrator:** explicit or semi‑implicit schemes with delay‑buffers; for memory surrogate use ODE solvers with adaptive steps and **energy monitors**.
- **Kernel ID:** generalized **Prony/Vector‑Fitting** on node/cluster impulse responses; enforce **stability & passivity**; report $L_2$ residuals and Hankel singular values.
- **Uncertainty:** bootstrap across disorder seeds and lattice realizations; report CIs on $\beta_\hbar$, $c$, $\hbar_\mathrm{eff}$.
- **LPC audit:** track $\mathcal{K}(t)$, chaos‑flux counters, and “brake” interventions; interventions must **decrease** $\mathcal{K}$ in closed subsystems.
- **Reproducibility artifacts:** CSVs for summaries, per‑run logs, and time‑series; JSON for parameters; seeds/version pins.

**Suggested CSV schema** (example):  
```
rid, N, a, tau0, gamma, xi, K, seed,
beta_hbar, beta_hbar_r2, hbar_eff_mean, hbar_eff_std,
c_est, c_ci_lo, c_ci_hi,
lpc_rate, lpc_violations, brake_count, anisotropy,
notes
```

---

## 10) Roadmap (6–12 months)

1. **Close $\hbar_\mathrm{eff}$ derivation** (stochastic invariant action under A0+A4′).  
2. **Constitutive $G$ map** from $\partial n_\mathrm{eff}^2/\partial \rho$ in micro → macro derivation.  
3. **Kernel rigor:** bounds for Prony/Vector‑Fitting approximation; passivity & error‑envelope theorems.  
4. **Self‑averaging campaign:** long runs with $d=2\ldots 64$; publish $\beta_\hbar$ maps vs. $\gamma,\xi,K$.  
5. **Analogue gravity experiment tie‑in:** predict $T_H$ slopes from extracted kernels; compare to BEC/optical data.  
6. **Atomic cross‑checks:** QDT fits on Na/Rb/Cs; verify predicted collapses and times $T_\mathrm{cl}, T_\mathrm{rev}$.

---

## 11) Glossary

- **Oscillator/node $q_i$:** primitive degree of freedom; identical units in a graph.  
- **Delay $\tau_{ij}$:** time for influence to traverse link $i\!\leftrightarrow\! j$; defines geometry.  
- **Holonomy $W(\ell)$:** gauge‑invariant loop phase; analogue of Wilson loop.  
- **Quantum floor $\xi$:** minimal stochastic drive; source of $\hbar_\mathrm{eff}$ in DOFT.  
- **Memory kernel $M$:** finite‑pole approximation of distributed delays.  
- **LPC:** Law of Preservation of Chaos; balance law for chaos content.  
- **Self‑averaging exponent $\beta_\hbar$:** scaling of relative variance of $\hbar_\mathrm{eff}$ with system size.  

---

## 12) Philosophy in one paragraph

Information‑bearing oscillations propagate in a **no‑background** substrate; **space** and **time** are the bookkeeping of finite influence speeds. **Order** is the persistent memory of past oscillations stabilized by dissipation; **chaos** is the conserved resource that powers emergence under the LPC. The physical world is the history of such oscillations coalescing into resilient patterns—atoms, fields, geometry—each a layered “skin” of coherence built atop the same simple rule: delayed interaction.

---

## 13) License & contributions

- **License:** Apache‑2.0 (suggested).  
- **Contributions:** please include (i) parameter JSONs, (ii) seeds, (iii) environment hashes, and (iv) raw/derived CSVs. Open an issue for proposals to extend the axiom set or to add new falsification tests.

---

*This is a living document. Tag releases as* `DOFT_Manifesto_v1.3`, *and cite component modules (kernel‑ID, LPC monitors, self‑averaging probes) with their semantic versions.*