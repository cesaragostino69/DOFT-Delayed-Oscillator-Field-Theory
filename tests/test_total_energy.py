import numpy as np
import pytest

from doft.models.model import DOFTModel, compute_total_energy


@pytest.fixture(params=[0.1, 0.5])
def coupling(request):
    return request.param


@pytest.fixture(params=[0.05, 0.2])
def damping(request):
    return request.param


@pytest.fixture(params=[
    np.array([0.3]),
    np.array([0.2, 0.4]),
])
def memory_params(request):
    return {"weights": request.param}


def test_total_energy_passive(coupling, damping, memory_params):
    model = DOFTModel(
        grid_size=4,
        a=coupling,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=damping,
        seed=0,
        dt_nondim=0.05,
        kernel_params=memory_params,
        max_ram_bytes=32 * 1024**3,
    )

    rng = np.random.default_rng(0)
    model.Q = rng.normal(scale=0.1, size=model.Q.shape)
    model.P = rng.normal(scale=0.1, size=model.P.shape)
    if model.y_states is not None:
        model.y_states[:] = 0.1

    model.Q_delay = model.Q.copy()
    model.last_energy = compute_total_energy(
        model.Q, model.P, model.a_nondim, model.y_states, model.kernel_params
    )

    initial_energy = model.last_energy
    energies = [initial_energy]
    for t_idx in range(10):
        model._step_imex(t_idx)
        e = compute_total_energy(
            model.Q, model.P, model.a_nondim, model.y_states, model.kernel_params
        )
        energies.append(e)

    assert max(energies) <= initial_energy + 1e-12
    assert min(energies) >= -1e-12
