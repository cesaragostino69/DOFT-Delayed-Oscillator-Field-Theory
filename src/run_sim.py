# src/run_sim.py
# -*- coding: utf-8 -*-
"""
Main entry point for running DOFT Phase 1 simulations.
This script orchestrates the execution of experiments defined in a JSON config file,
handles parallelization, and saves results according to the data contract.
(Version corrected to fix parameter parsing bug)
"""

import os
import json
import csv
import pathlib
import argparse
import uuid
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from .model import DOFTModel

def run_one_simulation(cfg, out_dir):
    """
    Executes a single simulation run for a given configuration.
    """
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    model = DOFTModel(cfg)

    experiment_type = cfg.get("experiment_type", "chaos")

    run_result = {}
    blocks_data = []

    if experiment_type == "pulse":
        run_result, blocks_data = model.run_pulse_experiment(xi_amp=float(cfg.get("xi_amp", 1e-4)))
    else: # Default to chaos experiment
        run_result, blocks_data = model.run_chaos_experiment(xi_amp=float(cfg.get("xi_amp", 1e-4)))

    # --- Data Contract Logging ---
    run_row = {
        "run_id": run_id,
        "seed": cfg["seed"],
        "a_mean": cfg["a"]["mean"],
        "tau_mean": cfg["tau0"]["mean"],
        "gamma": cfg["gamma"],
        "ceff_pulse": run_result.get("ceff_pulse"),
        "ceff_pulse_ic95": None,
        "anisotropy_max_pct": run_result.get("anisotropy_max_pct"),
        "lpc_deltaK_neg_frac": run_result.get("lpc_deltaK_neg_frac"),
        "lpc_vcount": run_result.get("lpc_vcount")
    }

    for block in blocks_data:
        block["run_id"] = run_id

    meta_info = { "run_id": run_id, "config": cfg }
    with open(out_dir / f"{run_id}_meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    return run_row, blocks_data

def main():
    parser = argparse.ArgumentParser(description="DOFT Phase 1 Simulation Runner")
    parser.add_argument("--config", required=True, help="Path to the JSON experiment configuration file.")
    parser.add_argument("--out", required=True, help="Output directory for results.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        config = json.load(f)

    # --- Build the list of simulation tasks from the config (Corrected Logic) ---
    tasks = []
    for experiment in config["experiments"]:
        # Keys that define a sweep of parameters
        sweep_keys = [k for k, v in experiment.items() if isinstance(v, list)]
        
        # Base configuration for this specific experiment
        exp_base_cfg = {k: v for k, v in config.items() if k != "experiments"}
        exp_base_cfg.update({k: v for k, v in experiment.items() if k not in sweep_keys})

        # Get the lists of values to sweep over
        sweep_values = [experiment[k] for k in sweep_keys]
        
        # Create the Cartesian product of all swept values
        for value_combination in itertools.product(*sweep_values):
            run_cfg = exp_base_cfg.copy()
            # Add the specific parameter combination for this run
            for i, key in enumerate(sweep_keys):
                # The key 'seeds' should be singular 'seed' for the model config
                run_key = 'seed' if key == 'seeds' else key
                run_cfg[run_key] = value_combination[i]
            tasks.append(run_cfg)

    print(f"# Starting {len(tasks)} simulation runs with {args.n_jobs} parallel jobs...")

    # --- Run simulations in parallel ---
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_one_simulation)(task_cfg, out_dir) for task_cfg in tqdm(tasks)
    )

    # --- Aggregate and save results ---
    all_runs = [r[0] for r in results if r is not None]
    all_blocks = [b for r in results if r is not None for b in r[1]]

    if all_runs:
        with open(out_dir / "runs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_runs[0].keys())
            writer.writeheader()
            writer.writerows(all_runs)
        print(f"# Saved {len(all_runs)} entries to runs.csv")

    if all_blocks:
        with open(out_dir / "blocks.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_blocks[0].keys())
            writer.writeheader()
            writer.writerows(all_blocks)
        print(f"# Saved {len(all_blocks)} entries to blocks.csv")

if __name__ == "__main__":
    main()
