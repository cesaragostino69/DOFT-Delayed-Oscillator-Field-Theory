import sys
from pathlib import Path
import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from doft.simulation import run_sim


def test_run_sim_leapfrog_validation(tmp_path, monkeypatch):
    """Ensure Leapfrog integrator with non-zero gamma raises ValueError."""
    monkeypatch.chdir(tmp_path)

    cfg = {
        'gamma': 0.1,
        'integrator': 'Leapfrog',
        'seeds': [0],
        'sweep_groups': {'g1': [[1.0, 1.0]]},
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setenv('DOFT_CONFIG', str(config_path))
    monkeypatch.setattr(sys, 'argv', ['run_sim'])

    with pytest.raises(ValueError):
        run_sim.main()
        