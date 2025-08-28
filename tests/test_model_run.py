# tests/test_model_run.py
"""Basic tests for the high-level ``DOFTModel.run`` interface."""

import sys
from pathlib import Path

import numpy as np

# Ensure the package import works when repository root is the current directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doft.models.model import DOFTModel


def create_model(seed=0, dt_nondim=0.1):
    return DOFTModel(
        grid_size=4,
        a=1.0,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=0.1,
        seed=seed,
        dt_nondim=dt_nondim,
    )


def test_pulse_metrics_numeric():
    model = create_model(seed=0)
    metrics = model._calculate_pulse_metrics(n_steps=50)

    assert "ceff_pulse" in metrics
    assert np.isfinite(metrics["ceff_pulse"])


def test_seed_reproducibility_lpc_metrics():
    m1 = create_model(seed=123)
    m2 = create_model(seed=123)
    r1, _ = m1._calculate_lpc_metrics(n_steps=50)
    r2, _ = m2._calculate_lpc_metrics(n_steps=50)
    assert r1 == r2
    