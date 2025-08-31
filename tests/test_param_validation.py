import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from doft.simulation import run_sim


def test_invalid_parameters(tmp_path, monkeypatch):
    cfg = {
        'seeds': [0],
        'sweep_groups': {'g1': [[1.0, 1.0]]},
        'epsilon_tau': 0.01,  # below allowed range
        'eta': 0.08,
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setenv('DOFT_CONFIG', str(config_path))
    monkeypatch.setattr(sys, 'argv', ['run_sim'])
    with pytest.raises(ValueError):
        run_sim.main()

    cfg['epsilon_tau'] = 0.1
    cfg['eta'] = 0.5  # above allowed range
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        run_sim.main()
        