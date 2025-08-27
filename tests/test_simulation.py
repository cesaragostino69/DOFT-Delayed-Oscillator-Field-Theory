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
