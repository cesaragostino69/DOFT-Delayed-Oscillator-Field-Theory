
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
=======
-   **A1 — Local Delayed Dynamics (DDE):** The fundamental equation of motion for a node `i` is a delayed differential equation:
  
    $$\frac{d^2q_i}{dt^2} + 2\gamma_i\frac{dq_i}{dt} + \omega_i^2q_i = \sum_j K_{ij} \sin( q_j(t - \tau_{ij}) - q_i(t) + A_{ij} ) + \xi_i(t)$$

### Prony memory kernel
=======
-   **A6 — Coarse-Graining with Memory:** Blocks of oscillators lead to a continuum wave-equation with a memory kernel `M`, implemented via a chain of ODE "memory variables".

    $$\frac{\partial^2\phi}{\partial t^2} - c^2\nabla^2\phi + \int_0^t M(t - t') \frac{\partial\phi}{\partial t'}(x,t') dt' = \eta(x,t)$$

