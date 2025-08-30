# src/doft/simulation/run_sim.py
import argparse
import logging
import pandas as pd
import numpy as np
import time
import json
import os
from pathlib import Path
import subprocess
import multiprocessing as mp

from doft.models.model import DOFTModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Globals for worker processes
_CONFIG = {}
_RESULTS = None
_COUNTER = None
_TOTAL = 0


def init_worker(config, results_list, counter, total):
    """Initializer for worker processes to set shared state."""
    global _CONFIG, _RESULTS, _COUNTER, _TOTAL
    _CONFIG = config
    _RESULTS = results_list
    _COUNTER = counter
    _TOTAL = total


def run_single_sim(a_val, tau_val, seed):
    """Run a single simulation and append results to the shared list."""
    with _COUNTER.get_lock():
        _COUNTER.value += 1
        run_idx = _COUNTER.value

    run_id = f"run_{int(time.time())}_{run_idx}"
    print(f"[{run_idx}/{_TOTAL}] Running sim: a={a_val}, τ={tau_val}, seed={seed}")

    model = DOFTModel(
        grid_size=_CONFIG['grid_size'],
        a=a_val,
        tau=tau_val,
        a_ref=_CONFIG['a_ref'],
        tau_ref=_CONFIG['tau_ref'],
        gamma=_CONFIG['gamma'],
        seed=seed,
        boundary_mode=_CONFIG['boundary_mode'],
        log_steps=_CONFIG['log_steps'],
        log_path=_CONFIG['log_path'],
        max_ram_bytes=_CONFIG['max_ram_bytes'],
        lpc_duration_physical=_CONFIG.get('lpc_duration_physical'),
    )

    run_metrics, blocks_df = model.run()
    logger.info(
        "run_id=%s C-1: ceff_pulse=%s ceff_pulse_ic95_lo=%s ceff_pulse_ic95_hi=%s "
        "C-2: var_c_over_c2=%s anisotropy_max_pct=%s "
        "C-3: lpc_ok_frac=%s lpc_vcount=%s",
        run_id,
        run_metrics.get('ceff_pulse'),
        run_metrics.get('ceff_pulse_ic95_lo'),
        run_metrics.get('ceff_pulse_ic95_hi'),
        run_metrics.get('var_c_over_c2'),
        run_metrics.get('anisotropy_max_pct'),
        run_metrics.get('lpc_ok_frac'),
        run_metrics.get('lpc_vcount'),
    )

    run_metrics['run_id'] = run_id
    run_metrics['seed'] = seed
    run_metrics['a_mean'] = a_val
    run_metrics['tau_mean'] = tau_val
    run_metrics['gamma'] = _CONFIG['gamma']
    run_metrics['param_group'] = _CONFIG['point_to_group'].get((a_val, tau_val), 'unknown')
    run_metrics['lorentz_window'] = 'NA'

    if blocks_df is not None and not blocks_df.empty:
        blocks_df['run_id'] = run_id
        if 'block_skipped' in blocks_df.columns:
            blocks_df['block_skipped'] = blocks_df['block_skipped'].astype(int)

    _RESULTS.append((run_metrics, blocks_df))

def main():
    """
    Main orchestrator for the DOFT Phase 1 counter-trial.
    This version incorporates numerical stability fixes based on audit feedback.
    """
    parser = argparse.ArgumentParser(description="Run DOFT Phase-1 Simulation Sweep.")
    parser.add_argument(
        "--config",
        help="Path to JSON configuration file. Overrides command line defaults.",
    )
    parser.add_argument(
        "--boundary",
        choices=["periodic", "reflective", "absorbing"],
        default="periodic",
        help="Boundary condition for lattice interactions",
    )
    parser.add_argument(
        "--log-steps",
        action="store_true",
        help="Persist per-step diagnostic metrics",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Prefix path for step log output files",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run simulations in parallel using multiprocessing",
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    config_path = args.config or os.environ.get('DOFT_CONFIG')
    cfg_json = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg_json = json.load(f)

    # general parameters with defaults
    seeds = cfg_json.get('seeds', [42, 123, 456, 789, 1011])
    gamma = cfg_json.get('gamma', 0.05)
    grid_size = cfg_json.get('grid_size', 100)
    boundary_mode = cfg_json.get('boundary_mode', args.boundary)
    log_steps = cfg_json.get('log_steps', args.log_steps)
    log_path = cfg_json.get('log_path', args.log_path)
    a_ref = cfg_json.get('a_ref', 1.0)
    tau_ref = cfg_json.get('tau_ref', 1.0)
    max_ram_bytes = cfg_json.get('max_ram_bytes', 32 * 1024**3)
    lpc_duration_physical = cfg_json.get('lpc_duration_physical')

    # --- Sweep Configuration ---
    simulation_points = []
    point_to_group = {}
    sweep_groups = cfg_json.get('sweep_groups')
    if sweep_groups:
        # Expect a mapping of group name -> list of [a, tau] pairs
        for name, pts in sweep_groups.items():
            for a_val, tau_val in pts:
                pt_t = (a_val, tau_val)
                simulation_points.append(pt_t)
                point_to_group[pt_t] = name
    else:
        # Default sweep configuration if none provided
        default_groups = {
            'g1': [(1.0, 1.0), (1.2, 1.2), (1.5, 1.5)],
            'g2': [(1.0, 1.0), (1.2, 1.0), (1.5, 1.0)],
            'g3': [(1.0, 1.0), (1.0, 0.8), (1.0, 0.67)],
        }
        for name, pts in default_groups.items():
            for pt in pts:
                simulation_points.append(pt)
                point_to_group[pt] = name

    # --- Create Unique Output Directory ---
    base_run_dir = 'runs'
    os.makedirs(base_run_dir, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_run_dir, f'phase1_run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Saving results to: {output_dir}")

    # --- Simulation Execution ---
    all_runs_data = []
    all_blocks_data = []

    print(f"🚀 Starting DOFT Phase-1 Simulation Sweep across {len(simulation_points)} points...")

    total_sims = len(simulation_points) * len(seeds)

    config = {
        'gamma': gamma,
        'grid_size': grid_size,
        'boundary_mode': boundary_mode,
        'log_steps': log_steps,
        'log_path': log_path,
        'a_ref': a_ref,
        'tau_ref': tau_ref,
        'point_to_group': point_to_group,
        'max_ram_bytes': max_ram_bytes,
        'lpc_duration_physical': lpc_duration_physical,
    }

    counter = mp.Value('i', 0)
    combos = [(a, t, s) for (a, t) in simulation_points for s in seeds]

    if args.parallel:
        with mp.Manager() as manager:
            results_list = manager.list()
            with mp.Pool(initializer=init_worker, initargs=(config, results_list, counter, total_sims)) as pool:
                pool.starmap(run_single_sim, combos)
            results = list(results_list)
    else:
        results = []
        init_worker(config, results, counter, total_sims)
        for args_tuple in combos:
            run_single_sim(*args_tuple)
        results = list(results)

    for run_metrics, blocks_df in results:
        all_runs_data.append(run_metrics)
        if blocks_df is not None and not blocks_df.empty:
            all_blocks_data.append(blocks_df)

    print(f"\n✅ Simulation sweep finished. Consolidating and writing results to {output_dir}...")

    runs_df = pd.DataFrame(all_runs_data)
    runs_output_path = os.path.join(output_dir, 'runs.csv')
    runs_df.to_csv(runs_output_path, index=False)
    print(f"--> Wrote {len(runs_df)} rows to {runs_output_path}")

    if all_blocks_data:
        blocks_df_final = pd.concat(all_blocks_data, ignore_index=True)
        blocks_output_path = os.path.join(output_dir, 'blocks.csv')
        blocks_df_final.to_csv(blocks_output_path, index=False)
        print(f"--> Wrote {len(blocks_df_final)} rows to {blocks_output_path}")
    else:
        print("--> No block data generated for blocks.csv.")

    meta_data = {
        'run_directory': f'phase1_run_{timestamp}',
        'timestamp_utc': time.asctime(time.gmtime()),
        'total_runs_in_sweep': total_sims,
        'simulation_points': simulation_points,
        'seeds_used': seeds,
        'fixed_params': {'gamma': gamma, 'grid_size': grid_size},
        'stability_params': {
            'dt_logic': 'min(0.02, 0.1, tau_nondim/50, 0.1/(gamma_nondim + |a_nondim| + 1))',
            'a_ref': a_ref,
            'tau_ref': tau_ref,
            'delay_interpolation': True,
        },
    }

    repo_root = Path(__file__).resolve().parents[2]
    try:
        code_version = subprocess.check_output([
            'git', 'rev-parse', 'HEAD'
        ], cwd=repo_root).decode().strip()
    except Exception:
        code_version = 'unknown'

    meta_data.update({
        'manifest': 'MANIFESTO.md',
        'code_version': code_version,
        'seeds_detailed': [{'seed': s} for s in seeds],
        'front_thresholds': [1.0, 3.0, 5.0],
    })
    meta_output_path = os.path.join(output_dir, 'run_meta.json')
    with open(meta_output_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
    print(f"--> Wrote metadata to {meta_output_path}")

if __name__ == "__main__":
    main()
