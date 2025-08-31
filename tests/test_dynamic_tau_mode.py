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
        max_pulse_steps=1,
        max_lpc_steps=1,
    )
    metrics, _ = model.run()
    assert metrics["tau_dynamic_on"] is True
    assert metrics["ring_buffer_len"] > 0