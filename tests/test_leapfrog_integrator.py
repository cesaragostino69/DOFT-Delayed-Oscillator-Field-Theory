# tests/test_leapfrog_integrator.py
"""Unit tests for the Leapfrog integrator."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package import works when repository root is the current directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doft.models.model import DOFTModel


def test_leapfrog_energy_conservation():
    """Leapfrog should conserve energy in the conservative regime."""

    model = DOFTModel(
        grid_size=4,
        a=1.0,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=0.0,
        seed=0,
        max_ram_bytes=32 * 1024**3,
    )

    rng = np.random.default_rng(0)
    model.Q = rng.normal(scale=0.1, size=model.Q.shape)
    model.P = rng.normal(scale=0.1, size=model.P.shape)
    model.last_energy = model.energy_fn(model.Q, model.P)
    initial_energy = model.last_energy

    for t_idx in range(100):
        model._step_leapfrog(t_idx)

    final_energy = model.energy_fn(model.Q, model.P)
    assert np.isfinite(final_energy)
    assert final_energy == pytest.approx(initial_energy, rel=2e-4, abs=1e-6)
