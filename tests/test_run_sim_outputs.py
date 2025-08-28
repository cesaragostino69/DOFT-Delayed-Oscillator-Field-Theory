import sys
from pathlib import Path
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from doft.simulation import run_sim


def test_run_sim_outputs(tmp_path, monkeypatch):
    # run simulations in a temporary directory to avoid polluting repo
    monkeypatch.chdir(tmp_path)

    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            metrics = {
                'xi_floor': 0.1,
                'ceff_pulse': 1.0,
                'ceff_pulse_ic95_lo': 0.9,
                'ceff_pulse_ic95_hi': 1.1,
                'anisotropy_max_pct': 0.0,
                'var_c_over_c2': 0.0,
                'ceff_iso_x': 1.0,
                'ceff_iso_y': 1.0,
                'ceff_iso_z': 1.0,
                'ceff_iso_diag': 1.0,
                'lpc_ok_frac': 1.0,
                'lpc_vcount': 0,
                'lpc_windows_analyzed': 1,
                'block_skipped': 0,
            }
            df = pd.DataFrame({
                'window_id': [0],
                'K_metric': [0.0],
                'deltaK': [0.0],
                'block_skipped': [0],
            })
            return metrics, df

    monkeypatch.setattr(run_sim, 'DOFTModel', DummyModel)
    monkeypatch.setattr(sys, 'argv', ['run_sim'])
    run_sim.main()

    run_dir = next((tmp_path / 'runs').glob('phase1_run_*'))
    runs_df = pd.read_csv(run_dir / 'runs.csv')
    required_cols = {'lpc_ok_frac', 'ceff_pulse_ic95_lo', 'ceff_pulse_ic95_hi', 'lorentz_window', 'ceff_iso_diag'}
    assert required_cols.issubset(runs_df.columns)

    with open(run_dir / 'run_meta.json') as f:
        meta = json.load(f)
    for field in ['manifest', 'code_version', 'seeds_detailed', 'front_thresholds']:
        assert field in meta
