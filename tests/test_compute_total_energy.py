import numpy as np

from doft.models.model import compute_energy, compute_total_energy


def test_total_energy_reduces_to_basic():
    Q = np.array([[1.0, 0.0], [0.0, -1.0]])
    P = np.zeros_like(Q)
    e_basic = compute_energy(Q, P)
    e_total = compute_total_energy(Q, P, K=0.0, y_states=None, kernel_params=None)
    assert np.isclose(e_total, e_basic)


def test_total_energy_with_coupling():
    Q = np.array([[0.0, 1.0], [1.0, 0.0]])
    P = np.zeros_like(Q)
    e_total = compute_total_energy(Q, P, K=1.0, y_states=None, kernel_params=None)
    # Expected: potential energy 1.0 + coupling 4.0
    assert np.isclose(e_total, 5.0)


def test_total_energy_with_memory():
    Q = np.zeros((2, 2))
    P = np.zeros_like(Q)
    y = np.ones((2, 2, 2))
    params = {"weights": np.array([0.5, 1.5])}
    e_total = compute_total_energy(Q, P, K=0.0, y_states=y, kernel_params=params)
    # Each mode contributes 0.5 * weight * 4
    assert np.isclose(e_total, 4.0)
