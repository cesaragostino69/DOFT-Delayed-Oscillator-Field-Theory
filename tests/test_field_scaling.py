# tests/test_field_scaling.py
import sys
from pathlib import Path

import numpy as np

# Ensure the package import works when repository root is the current directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doft.models.model import DOFTModel


def test_field_scaling_applied():
    model = DOFTModel(
        grid_size=4,
        a=1.0,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=0.1,
        seed=0,
    )

    large_value = model.scale_threshold * 2
    model.Q.fill(large_value)
    model.P.fill(large_value)
    model.Q_history.fill(large_value)
    model.last_energy = float(0.5 * np.sum(model.P**2) + 0.5 * np.sum(model.Q**2))
    energy_before = model.last_energy

    model._step_euler(0)

    assert model.scale_accum > 1.0
    assert np.linalg.norm(model.Q) <= model.scale_threshold
    assert np.linalg.norm(model.P) <= model.scale_threshold
    assert model.scale_log[-1] == model.scale_accum
    idx = (1) % model.history_steps
    assert np.allclose(model.Q_history[idx] * model.scale_accum, large_value)
    assert np.isclose(model.last_energy, energy_before / model.scale_accum**2, rtol=1e-2)
    