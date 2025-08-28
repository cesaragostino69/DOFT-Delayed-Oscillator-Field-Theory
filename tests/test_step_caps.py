import sys
from pathlib import Path

import pandas as pd

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from doft.models.model import DOFTModel


class RecordingModel(DOFTModel):
    """Subclass that records the number of steps requested."""

    def _calculate_pulse_metrics(self, n_steps, noise_std: float = 0.0):
        self.recorded_pulse_steps = n_steps
        return {}

    def _calculate_lpc_metrics(self, n_steps):
        self.recorded_lpc_steps = n_steps
        return {}, pd.DataFrame()


def test_run_respects_step_caps():
    # Choose parameters that make dt extremely small so computed steps are huge
    model = RecordingModel(
        grid_size=4,
        a=1.0,
        tau=0.01,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=0.1,
        seed=0,
        max_pulse_steps=50,
        max_lpc_steps=60,
    )

    # Sanity check: without caps the computed steps would be much larger
    assert int(3000 * (0.1 / model.dt)) > 50
    assert int(30000 * (0.1 / model.dt)) > 60

    model.run()

    assert model.recorded_pulse_steps == 50
    assert model.recorded_lpc_steps == 60
    