# tests/test_boundary_modes.py
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure package import works when repository root is current directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doft.models.model import DOFTModel


def create_model(boundary):
    return DOFTModel(
        grid_size=3,
        a=1.0,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=0.0,
        seed=0,
        boundary_mode=boundary,
        dt_nondim=0.1,
    )


def test_periodic_wraparound():
    m = create_model("periodic")
    field = np.zeros((3, 3))
    field[0, 0] = 1.0
    lap = m._laplacian(field)
    assert lap[2, 0] == pytest.approx(1.0)


def test_absorbing_zero_padding():
    m = create_model("absorbing")
    field = np.zeros((3, 3))
    field[0, 0] = 1.0
    lap = m._laplacian(field)
    assert lap[2, 0] == pytest.approx(0.0)
    assert lap[0, 0] == pytest.approx(-4.0)


def test_reflective_boundary():
    m = create_model("reflective")
    field = np.zeros((3, 3))
    field[0, 1] = 1.0
    lap = m._laplacian(field)
    assert lap[0, 1] == pytest.approx(-3.0)
