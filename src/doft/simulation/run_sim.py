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
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - fallback if psutil not installed
    psutil = None
    import resource

from doft.models.model import DOFTModel, DEFAULT_MAX_RAM_BYTES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Globals for worker processes
_CONFIG = {}
_RESULTS = None
_COUNTER = None
_TOTAL = 0


def _load_env_config():
    """Load simulation configuration from the ``DOFT_CONFIG`` environment variable.

    The variable may contain either a JSON string or a path to a JSON file.
    Returns an empty dict if the variable is unset.
    """
    cfg_raw = os.environ.get("DOFT_CONFIG")
    if not cfg_raw:
        return {}
    try:
        if os.path.exists(cfg_raw):
            with open(cfg_raw, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(cfg_raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Invalid DOFT_CONFIG: {exc}")


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

    try:
        model = DOFTModel(
            grid_size=_CONFIG['grid_size'],
            a=a_val,
            tau=tau_val,
            a_ref=_CONFIG['a_ref'],
            tau_ref=_CONFIG['tau_ref'],
            gamma=_CONFIG['gamma'],
            seed=seed,
            boundary_mode=_CONFIG['boundary_mode'],
            dt_nondim=_CONFIG.get('dt_nondim'),
            max_pulse_steps=_CONFIG.get('max_pulse_steps'),
            max_lpc_steps=_CONFIG.get('max_lpc_steps'),
            kernel_params=_CONFIG.get('kernel_params'),
            energy_mode=_CONFIG.get('energy_mode', 'auto'),
            log_steps=_CONFIG['log_steps'],
            log_path=_CONFIG['log_path'],
            max_ram_bytes=_CONFIG.get('mem_limit_bytes'),
        )
        run_metrics, blocks_df = model.run()
    except MemoryError as e:
        msg = f"MemoryError in {run_id}: {e}"
        logger.error(msg)
        print(msg)
        return

    if psutil is not None:
        mem_usage = psutil.Process(os.getpid()).memory_info().rss
    else:  # pragma: no cover - platform dependent fallback
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    logger.info("run_id=%s memory_mb=%.2f", run_id, mem_usage / (1024 ** 2))
    limit = _CONFIG.get('mem_limit_bytes')
    if limit and mem_usage > 0.9 * limit:
        logger.warning(
            "run_id=%s memory usage %.2f MB approaching limit %.2f MB",
            run_id,
            mem_usage / (1024 ** 2),
            limit / (1024 ** 2),
        )
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
    This version uses the new IMEX/Leapfrog integrator with Prony memory.
    """
    parser = argparse.ArgumentParser(description="Run DOFT Phase-1 Simulation Sweep.")
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
<<<<<<< ours
    parser.add_argument(
        "--mem-limit",
        type=float,
        default=None,
        help="Soft memory limit in GB; warnings emitted when 90% is reached",
    )
    args = parser.parse_args()

    env_cfg = _load_env_config()
    default_cfg = {
        "gamma": 0.05,
        "grid_size": 100,
        "a_ref": 1.0,
        "tau_ref": 1.0,
        "groups": {
            "g1": [(1.0, 1.0), (1.2, 1.2), (1.5, 1.5)],
            "g2": [(1.0, 1.0), (1.2, 1.0), (1.5, 1.0)],
            "g3": [(1.0, 1.0), (1.0, 0.8), (1.0, 0.67)],
        },
        "seeds": [42, 123, 456, 789, 1011],
    }
    cfg = {**default_cfg, **env_cfg}

    # Determine simulation points and grouping
    point_to_group = {}
    if "groups" in cfg:
        simulation_points = []
        for gname, pts in cfg["groups"].items():
            for pt in pts:
                pt_tuple = tuple(pt)
                simulation_points.append(pt_tuple)
                point_to_group[pt_tuple] = gname
    else:
        simulation_points = [tuple(pt) for pt in cfg.get("simulation_points", [])]
        point_to_group = {}

    seeds = cfg.get("seeds", [])
    gamma = cfg.get("gamma", 0.05)
    grid_size = cfg.get("grid_size", 100)
    a_ref = cfg.get("a_ref", 1.0)
    tau_ref = cfg.get("tau_ref", 1.0)

    boundary_mode = cfg.get("boundary_mode", args.boundary)
    log_steps = cfg.get("log_steps", args.log_steps)
    log_path = cfg.get("log_path", args.log_path)
    dt_nondim = cfg.get("dt_nondim")
    max_pulse_steps = cfg.get("max_pulse_steps")
    max_lpc_steps = cfg.get("max_lpc_steps")
    kernel_params = cfg.get("kernel_params")
    energy_mode = cfg.get("energy_mode", "auto")
=======
    repo_root = Path(__file__).resolve().parents[2]
    default_config = os.environ.get("DOFT_CONFIG") or str(
        repo_root / "configs" / "config_phase1.json"
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to JSON configuration file",
    )
    args = parser.parse_args()

    # --- Load Config ---
    with open(args.config, "r") as f:
        cfg = json.load(f)

    gamma = cfg.get("gamma", 0.05)
    grid_size = cfg.get("grid_size", 100)
    seeds = cfg.get("seeds", [42, 123, 456, 789, 1011])
    sweep_groups = cfg.get("sweep_groups", {})

    simulation_points = []
    point_to_group = {}
    for group_name, points in sweep_groups.items():
        for pt in points:
            pt_tuple = tuple(pt)
            simulation_points.append(pt_tuple)
            point_to_group[pt_tuple] = group_name

    # STABILITY FIX: Define reference parameters for nondimensionalization.
    # We use the central point of the sweep as the reference scale.
    a_ref = 1.0
    tau_ref = 1.0
>>>>>>> theirs

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

    mem_limit_bytes = (
        int(args.mem_limit * 1024 ** 3)
        if args.mem_limit is not None
        else DEFAULT_MAX_RAM_BYTES
    )

    config = {
        'gamma': gamma,
        'grid_size': grid_size,
        'boundary_mode': boundary_mode,
        'log_steps': log_steps,
        'log_path': log_path,
        'a_ref': a_ref,
        'tau_ref': tau_ref,
        'point_to_group': point_to_group,
        'mem_limit_bytes': mem_limit_bytes,
        'dt_nondim': dt_nondim,
        'max_pulse_steps': max_pulse_steps,
        'max_lpc_steps': max_lpc_steps,
        'kernel_params': kernel_params,
        'energy_mode': energy_mode,
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
        'sweep_groups': sweep_groups,
        'config_file': args.config,
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
