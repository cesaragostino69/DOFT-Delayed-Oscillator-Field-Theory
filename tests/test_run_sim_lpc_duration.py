import sys
from pathlib import Path
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from doft.simulation import run_sim

def test_run_sim_respects_lpc_duration(tmp_path, monkeypatch):
    """Run ``run_sim.main`` with a minimal JSON config and verify that
    ``lpc_duration_physical`` is forwarded to ``DOFTModel`` and that
    results are written to disk."""

    # run simulations in a temporary directory
    monkeypatch.chdir(tmp_path)

    captured = {}

    class DummyModel:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self):
            metrics = {
                'ceff_pulse': 1.0,
                'ceff_pulse_ic95_lo': 0.9,
                'ceff_pulse_ic95_hi': 1.1,
                'lpc_ok_frac': 1.0,
                'lpc_vcount': 0,
                'lpc_windows_analyzed': 1,
                'dt_max_delta_d_exceeded_count': 0,
                'delta_d_rate': 0.0,
            }
            df = pd.DataFrame({'window_id': [0], 'K_metric': [0.0],
                               'deltaK': [0.0], 'block_skipped': [0]})
            return metrics, df

    monkeypatch.setattr(run_sim, 'DOFTModel', DummyModel)

    cfg = {
        # Minimal settings plus ``lpc_duration_physical``
        'lpc_duration_physical': 2.0,
        'seeds': [0],
        'sweep_groups': {'g': [[1.0, 1.0]]},
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setenv('DOFT_CONFIG', str(config_path))
    monkeypatch.setattr(sys, 'argv', ['run_sim'])

    run_sim.main()

    # Ensure the parameter was propagated to the model
    assert captured['lpc_duration_physical'] == cfg['lpc_duration_physical']

    # Ensure results were written to disk
    run_dir = next((tmp_path / 'runs' / 'passive').glob('phase1_run_*'))
    assert (run_dir / 'runs.csv').exists()
