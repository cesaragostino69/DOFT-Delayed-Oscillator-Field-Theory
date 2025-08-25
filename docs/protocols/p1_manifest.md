Protocol 1: hbar_eff Self-Averaging
1. Objective
This protocol aims to test Axiom A4' (Emergent Quantum Floor) by measuring the self-averaging exponent β_h. We will verify if β_h approaches 1 in a homogeneous noise regime.

2. Methodology
We will perform multiple simulation runs with varying noise amplitudes (xi_amp) and dissipation factors (gamma). For each run, we will compute hbar_eff by block-averaging over different mesh scales and fit the relative variance to R ~ N^(-β_h).

3. Registered Runs
This table links each experimental run to the exact code version and configuration used. The results are stored locally in the /results/protocol_1/ directory (which is not tracked by Git).

Run ID

Code Version

Configuration File

Description

run_1

v1.0

configs/protocol_1/p1_run1_quick.yml

Quick smoke test with low step count.

run_2

v1.0

configs/protocol_1/p1_run2_long.yml

Long run with high precision for publication.

run_3

v1.1

configs/protocol_1/p1_run3_long.yml

Re-run of run_2 with updated integrator.


