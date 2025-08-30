# tests/test_integrator_stability.py
"""Unit tests for the semi-implicit integrator stability.

These tests instantiate the ``DOFTModel`` with increasingly larger
dimensionless time steps (``dt_nondim``) and ensure that the integration
remains numerically stable: all state variables stay finite and do not
explode after a modest number of iterations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package import works when repository root is the current directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doft.models.model import DOFTModel


@pytest.mark.parametrize("dt_nondim", [0.005, 0.05, 0.1, 0.5])
def test_semiimplicit_integrator_stability(dt_nondim):
    """The updated integrator should remain stable for larger time steps."""

    model = DOFTModel(
        grid_size=4,
        a=1.0,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=0.1,
        seed=0,
        dt_nondim=dt_nondim,
        max_ram_bytes=32 * 1024**3,
    )

    # Provide a small perturbation to avoid trivial dynamics
    model.Q = model.rng.normal(0, 0.1, model.Q.shape)

    for t_idx in range(20):
        model._step_euler(t_idx)

    assert np.all(np.isfinite(model.Q))
    assert np.all(np.isfinite(model.P))
    # The solution should remain bounded even for relatively large dt
    assert np.max(np.abs(model.Q)) < 1e3
