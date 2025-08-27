# tests/test_simulation.py
"""Basic integration test for DOFTModel.run"""
import math
import sys
from pathlib import Path

import numpy as np

# Ensure the package import works when repository root is the current directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doft.models.model import DOFTModel


import pytest


@pytest.mark.parametrize("gamma, omega", [(0.1, 0.5), (0.2, 1.0)])
def test_doft_model_run_returns_numeric_results(tmp_path, gamma, omega):
    cfg = {
        "steps": 5,
        "dt": 0.01,
        "K": 0.3,
        "seed": 0,
        "batch_replicas": 1,
        "L": 4,
        "a": {"mean": 1.0, "sigma": 0.1},
        "tau0": {"mean": 1.0, "sigma": 0.1},
        "gamma": gamma,
        "omega": omega,
    }

    model = DOFTModel(cfg)
    results, _ = model.run(xi_amp=0.01, seed=0, out_dir=str(tmp_path))

    assert isinstance(results, dict)
    assert "ceff_pulse" in results and "anisotropy_max_pct" in results
    assert "hbar_eff" in results and "lpc_rate" in results
    for key in ("ceff_pulse", "anisotropy_max_pct", "hbar_eff", "lpc_rate"):
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
        "gamma": 0.15,
        "omega": 0.9,
    }

    model = DOFTModel(cfg)
    center = model.L // 2
    for i in range(model.L):
        for j in range(model.L):
            r = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            model.Q[i, j] = max(0.0, 1.0 - 0.2 * r)

    results, _ = model.run(xi_amp=0.05, seed=0, out_dir=str(tmp_path))
    ceff = results["ceff_pulse"]
    assert ceff > 0 and math.isfinite(ceff)


def test_hbar_eff_and_lpc_rate_sensitive_to_gamma(tmp_path):
    base = {
        "steps": 10,
        "dt": 0.1,
        "K": 0.3,
        "seed": 0,
        "batch_replicas": 1,
        "L": 4,
        "a": {"mean": 1.0, "sigma": 0.1},
        "tau0": {"mean": 1.0, "sigma": 0.1},
        "omega": 0.9,
    }

    cfg1 = dict(base); cfg1["gamma"] = 0.05
    cfg2 = dict(base); cfg2["gamma"] = 0.25

    model1 = DOFTModel(cfg1)
    res1, _ = model1.run(xi_amp=0.01, seed=0, out_dir=str(tmp_path / "r1"))
    model2 = DOFTModel(cfg2)
    res2, _ = model2.run(xi_amp=0.01, seed=0, out_dir=str(tmp_path / "r2"))

    assert res1["hbar_eff"] != res2["hbar_eff"]
    assert res1["lpc_rate"] != res2["lpc_rate"]


def test_anisotropy_max_pct_detects_map_diff(tmp_path):
    cfg = {
        "steps": 60,
        "dt": 0.1,
        "K": 0.3,
        "seed": 0,
        "batch_replicas": 1,
        "L": 16,
        "a": {"mean": 1.0, "sigma": 0.1},
        "tau0": {"mean": 1.0, "sigma": 0.1},
        "gamma": 0.15,
        "omega": 0.9,
    }

    model = DOFTModel(cfg)
    center = model.L // 2
    model.Q.fill(0.0)
    model.Q[center, center] = 1.0
    model.tau_map = np.ones_like(model.tau_map) * 5.0
    model.tau_map[center, :] = 1.0
    model.tau_steps = np.ceil(model.tau_map / model.dt).astype(int)
    model.win = int(model.tau_steps.max()) + 1
    model.bufQ = np.zeros((model.L, model.L, model.win), dtype=np.float64)
    model.ceff_map = model.a_map / (model.tau_map + 1e-12)
    results, _ = model.run(xi_amp=0.05, seed=0, out_dir=str(tmp_path))

    assert results["anisotropy_max_pct"] > 0


def test_seed_reproducibility(tmp_path):
    cfg = {
        "steps": 5,
        "dt": 0.1,
        "K": 0.3,
        "seed": 0,
        "batch_replicas": 1,
        "L": 4,
        "a": {"mean": 1.0, "sigma": 0.1},
        "tau0": {"mean": 1.0, "sigma": 0.1},
        "gamma": 0.1,
        "omega": 0.5,
    }

    m1 = DOFTModel(cfg)
    m2 = DOFTModel(cfg)
    res1, _ = m1.run(xi_amp=0.01, seed=123, out_dir=str(tmp_path / "r1"))
    res2, _ = m2.run(xi_amp=0.01, seed=123, out_dir=str(tmp_path / "r2"))
    assert res1 == res2

    m3 = DOFTModel(cfg)
    res3, _ = m3.run(xi_amp=0.01, seed=321, out_dir=str(tmp_path / "r3"))
    assert res1 != res3
