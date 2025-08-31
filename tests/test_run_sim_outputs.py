import sys
from pathlib import Path
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from doft.simulation import run_sim


def test_run_sim_outputs(tmp_path, monkeypatch):
    # run simulations in a temporary directory to avoid polluting repo
    monkeypatch.chdir(tmp_path)

    captured = {}

    class DummyModel:
        def __init__(self, *args, **kwargs):
            # capture all initialization kwargs so we can ensure run_sim passes
            # the values it read from the configuration file
            captured.update(kwargs)

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
                'dt_max_delta_d_exceeded_count': 0,
                'delta_d_rate': 0.0,
            }
            df = pd.DataFrame({
                'window_id': [0],
                'K_metric': [0.0],
                'deltaK': [0.0],
                'block_skipped': [0],
            })
            return metrics, df

    monkeypatch.setattr(run_sim, 'DOFTModel', DummyModel)

    cfg = {
        # Use non-default values to ensure they are picked up from the config
        'a_ref': 2.0,
        'tau_ref': 3.0,
        'gamma': 0.05,
        'grid_size': 4,
        'boundary_mode': 'absorbing',
        'log_steps': True,
        'log_path': 'my_logs',
        'lpc_duration_physical': 1.0,
        'pulse_amplitude': 0.2,
        'detection_thresholds': [2.0, 4.0],
        'seeds': [0],
        'sweep_groups': {
            'g1': [[1.0, 1.0]],
        },
        'integrator': 'IMEX',
        'tau_model': 'z_aug',
        'epsilon_tau': 0.15,
        'eta': 0.08,
        'alpha_delay': 0.2,
        'lambda_z': 0.3,
        'tau_dynamic_on': True,
        'prony_memory': {'weights': [0.1], 'thetas': [0.2]},
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setenv('DOFT_CONFIG', str(config_path))
    monkeypatch.setattr(sys, 'argv', ['run_sim'])
    run_sim.main()

    # verify that configuration parameters were passed through
    assert captured['a_ref'] == cfg['a_ref']
    assert captured['tau_ref'] == cfg['tau_ref']
    assert captured['boundary_mode'] == cfg['boundary_mode']
    assert captured['log_steps'] == cfg['log_steps']
    assert captured['log_path'] == cfg['log_path']
    assert captured['pulse_amplitude'] == cfg['pulse_amplitude']
    assert captured['detection_thresholds'] == cfg['detection_thresholds']
    assert captured['integrator'] == 'IMEX'
    assert captured['alpha_delay'] == cfg['alpha_delay']
    assert captured['lambda_z'] == cfg['lambda_z']
    assert captured['epsilon_tau'] == cfg['epsilon_tau']
    assert captured['eta_slew'] == cfg['eta']
    assert captured['kernel_params'] == cfg['prony_memory']

    run_dir = next((tmp_path / 'runs' / 'passive').glob('phase1_run_*'))
    runs_df = pd.read_csv(run_dir / 'runs.csv')
    required_cols = {'lpc_ok_frac', 'ceff_pulse_ic95_lo', 'ceff_pulse_ic95_hi', 'lorentz_window', 'ceff_iso_diag', 'delta_d_rate'}
    assert required_cols.issubset(runs_df.columns)

    with open(run_dir / 'run_meta.json') as f:
        meta = json.load(f)
    for field in [
        'manifest',
        'code_version',
        'seeds_detailed',
        'detection_thresholds',
        'pulse_amplitude',
        'tau_model',
        'epsilon_tau',
        'eta',
        'alpha_delay',
        'lambda_z',
        'topology',
    ]:
        assert field in meta
    assert meta['tau_model'] == cfg['tau_model']
    assert meta['epsilon_tau'] == cfg['epsilon_tau']
    assert meta['eta'] == cfg['eta']
    assert meta['alpha_delay'] == cfg['alpha_delay']
    assert meta['lambda_z'] == cfg['lambda_z']
    assert meta['topology']['boundary_mode'] == cfg['boundary_mode']
    assert meta['topology']['grid'] == [cfg['grid_size'], cfg['grid_size']]
