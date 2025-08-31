import pytest
from doft.models.model import DOFTModel

def _make_model(params):
    return DOFTModel(
        grid_size=4,
        a=0.0,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=0.1,
        seed=0,
        kernel_params=params,
        max_ram_bytes=32 * 1024**3,
    )

def test_negative_weight_raises():
    params = {"weights": [0.1, -0.2], "thetas": [0.1, 0.2]}
    with pytest.raises(ValueError):
        _make_model(params)

def test_nonpositive_theta_raises():
    params = {"weights": [0.1], "thetas": [0.0]}
    with pytest.raises(ValueError):
        _make_model(params)
        