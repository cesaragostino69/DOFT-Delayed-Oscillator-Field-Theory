# tests/test_simulation.py
"""Basic integration test for DOFTModel.run"""
import math
import sys
from pathlib import Path

import numpy as np

# Ensure the package import works when repository root is the current directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doft.models.model import DOFTModel


def test_doft_model_run_returns_numeric_results(tmp_path):
    cfg = {
        "steps": 5,
        "dt": 0.01,
        "K": 0.3,
        "seed": 0,
        "batch_replicas": 1,
        "L": 4,
        "a": {"mean": 1.0, "sigma": 0.1},
        "tau0": {"mean": 1.0, "sigma": 0.1},
    }

    model = DOFTModel(cfg)
    results, _ = model.run(gamma=0.1, xi_amp=0.01, seed=0, out_dir=str(tmp_path))

    assert isinstance(results, dict)
    assert "ceff_pulse" in results and "anisotropy_max_pct" in results
    for key in ("ceff_pulse", "anisotropy_max_pct"):
        val = results[key]
        assert not (isinstance(val, float) and math.isnan(val))


def test_run_returns_positive_ceff_when_threshold_crossed(tmp_path):
    cfg = {
        "steps": 50,
        "dt": 0.1,
        "K": 0.3,
        "seed": 0,
        "batch_replicas": 1,
        "L": 16,
        "a": {"mean": 1.0, "sigma": 0.1},
        "tau0": {"mean": 1.0, "sigma": 0.1},
    }

    model = DOFTModel(cfg)
    center = model.L // 2
    for i in range(model.L):
        for j in range(model.L):
            r = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            model.Q[i, j] = max(0.0, 1.0 - 0.2 * r)

    results, _ = model.run(gamma=0.1, xi_amp=0.05, seed=0, out_dir=str(tmp_path))
    ceff = results["ceff_pulse"]
    assert ceff > 0 and math.isfinite(ceff)
