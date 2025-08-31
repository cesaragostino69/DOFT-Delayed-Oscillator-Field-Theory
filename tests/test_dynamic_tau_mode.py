# tests/test_dynamic_tau_mode.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doft.models.model import DOFTModel


def test_dynamic_tau_logging():
    model = DOFTModel(
        grid_size=5,
        a=0.1,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=0.0,
        seed=0,
        tau_dynamic=True,
        interp_order=5,
        max_pulse_steps=1,
        max_lpc_steps=1,
    )

    tau = model._compute_dynamic_tau()
    _, delay_steps, delta_d = model._get_delayed_q_interpolated(tau)
    assert delay_steps.shape == model.Q.shape
    assert model.delta_d_log[-1] == delta_d

    metrics, _ = model.run()
    assert metrics["tau_dynamic_on"] is True
    assert metrics["ring_buffer_len"] > 0
    assert model.interp_order == 5
    assert len(model.delta_d_log) > 0
