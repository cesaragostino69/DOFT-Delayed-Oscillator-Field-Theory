# DOFT — Delayed Oscillator Field Theory
*Emergent space-time, spectra, and gravity from state-dependent delays*

**Status:** Research Alpha | Open Methodology | Cross-Lab Collaboration

---

## 1. Overview

This repository hosts the reference implementation, experiments, and analysis pipeline for DOFT—a bottom-up, network-dynamics framework where the building blocks are identical oscillators coupled through retarded links. In DOFT, **space, time, fields, and gravitation are not fundamental objects**; they emerge from the causal pattern of delays and phases on a large graph.

The project's goals are:
-   **Theory-to-data:** Derive falsifiable predictions (scalings, collapse laws, stability bounds) from DOFT’s axioms.
-   **Data-to-theory:** Test those predictions with numerics and public datasets, and report success/failure with code-audited, reproducible runs.

## 2. Core Dynamics (Axioms)

The theory is built upon a set of core axioms that define the microscopic ontology and dynamics of the system.

-   **A1 — Local Delayed Dynamics (DDE):** The fundamental equation of motion for a node `i` is a delayed differential equation:
    $$
    \frac{d^2q_i}{dt^2} + 2\gamma_i\frac{dq_i}{dt} + \omega_i^2q_i = \sum_j K_{ij} \sin( q_j(t - \tau_{ij}) - q_i(t) + A_{ij} ) + \xi_i(t)
    $$

-   **A2 — Space-Time as a Delay Map:** The set of delays `{τ_ij}` encodes the causal structure. No background geometry is assumed.

-   **A3 — Gauge/Holonomy Principle:** Only loop phases (holonomies) are observable; local phases are gauge.

-   **A4′ — Emergent Quantum Floor:** An irreducible action scale, `ħ_eff`, arises statistically from the network’s chaotic, delayed dynamics; it is not imposed.

-   **A5 — Gravity from Delay Gradients:** Coarse gradients of `τ(x)` act as an effective refractive index, causing rays to curve and clocks to dilate.

-   **A6 — Coarse-Graining with Memory:** Blocks of oscillators lead to a continuum wave-equation with a memory kernel `M`, implemented via a chain of ODE "memory variables".
    $$
    \frac{\partial^2\phi}{\partial t^2} - c^2\nabla^2\phi + \int_0^t M(t - t') \frac{\partial\phi}{\partial t'}(x,t') dt' = \eta(x,t)
    $$

-   **LPC — Law of Preservation of Chaos:** The chaos budget is channeled and dissipated; the chaos functional does not grow without bounds under closed dynamics.

## 3. Key Experiments & Metrics

We test the theory by measuring specific, emergent quantities and comparing them against predictions.

-   **`hbar_eff` Self-Averaging:**
    -   **Quantity:** `hbar_eff ∝ σ_Q * σ_P`.
    -   **Test:** Block-averaging across mesh scales. We fit the relative variance `R ~ N^(-β_h)`.
    -   **Prediction:** `β_h ≈ 1` indicates strong self-averaging.

-   **Isotropy and Stability:**
    -   **Quantity:** `(Δc)/c = |c_x - c_y| / c_bar`.
    -   **Expectation:** `(Δc)/c ≪ 10⁻³` after transients, with zero LPC violations in healthy runs.

-   **Spectral Mapping (Rydberg/QDT):**
    -   **Test:** Collapse transition frequencies to find classical and revival times.
    -   **Interpretation:** Short-loop holonomies map onto Quantum Defect Theory (QDT) defects `δ_l`.

## 4. Repository Structure

-   **/configs**: YAML/JSON configuration files for simulation runs.
-   **/data**: Small reference datasets.
-   **/docs**: Whitepapers, design notes, and mathematical appendices.
-   **/notebooks**: Jupyter Notebooks for analysis and visualization.
-   **/results**: (Ignored by Git) Output directory for simulation artifacts.
-   **/scripts**: Utility scripts for common tasks.
-   **/src/doft**: Main Python source code for the simulator.
-   **/theory**: Axioms, derivations, and the falsification plan.
-   **/ci**: Static checks, unit tests, and audit scripts.

## 5. Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone <REPO_URL>
    cd <REPO_NAME>
    ```

2.  **Create and Activate Environment:** We recommend Conda for managing the environment.
    ```bash
    conda env create -f environment.yml
    conda activate doft_v12
    ```

3.  **(Optional, for GPU) Install PyTorch Wheels:**
    ```bash
    pip install --no-deps --force-reinstall /path/to/torch.whl /path/to/torchvision.whl
    ```

4.  **Run a Quick Smoke Test:**
    ```bash
    # For CPU
    USE_GPU=0 python -m src.run_sim --config configs/quick_test.yml --out results/smoke_test_cpu

    # For GPU
    USE_GPU=1 python -m src.run_sim --config configs/quick_test.yml --out results/smoke_test_gpu
    ```

## 6. Workflow & Governance

This project follows a strict, multi-party workflow to ensure correctness and reproducibility.

-   **Evaluators (OpenAI/Google):** Propose experiments and acceptance criteria via Pull Requests to the `/configs/` directory.
-   **Developer (Google Track):** Implements features and solvers, including unit tests and performance notes.
-   **Code Auditor (OpenAI Track):** Reviews numerical stability, determinism, and metric integrity. Has the authority to block merges that fail audit.
-   **Runner:** Executes merged experiments and publishes signed artifacts to a results store.
