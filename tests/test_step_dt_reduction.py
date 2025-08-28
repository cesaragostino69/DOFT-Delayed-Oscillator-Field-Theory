# tests/test_step_dt_reduction.py
import sys
from pathlib import Path

import numpy as np

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doft.models.model import DOFTModel


def test_dt_reduction_on_nonfinite():
    model = DOFTModel(
        grid_size=4,
        a=1.0,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=-10.0,
        seed=0,
        dt_nondim=0.1,
    )

    model.Q = model.rng.normal(0, 0.1, model.Q.shape)

    model._step_euler(0)

    assert np.all(np.isfinite(model.Q))
    assert np.all(np.isfinite(model.P))
    assert model.dt_nondim < 0.1


def test_dt_reduction_on_energy_increase():
    model = DOFTModel(
        grid_size=4,
        a=1.0,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=-0.1,
        seed=0,
        dt_nondim=0.1,
    )

    model.Q = model.rng.normal(0, 0.1, model.Q.shape)

    initial_dt = model.dt_nondim
    model._step_euler(0)

    assert model.dt_nondim < initial_dt
