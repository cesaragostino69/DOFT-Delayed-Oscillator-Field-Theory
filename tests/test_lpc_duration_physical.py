import sys
from pathlib import Path
import math

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from doft.models.model import DOFTModel


def test_lpc_duration_sets_max_steps():
    duration = 1.0
    model = DOFTModel(
        grid_size=4,
        a=1.0,
        tau=1.0,
        a_ref=1.0,
        tau_ref=1.0,
        gamma=0.1,
        seed=0,
        lpc_duration_physical=duration,
        max_ram_bytes=32 * 1024**3,
    )

    expected = math.ceil(duration / model.dt)
    assert model.max_lpc_steps == expected
    