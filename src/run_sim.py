# src/doft/run_sim.py
# -*- coding: utf-8 -*-
"""
Main entry point for running DOFT Phase 1 simulations.
This script orchestrates the execution of experiments defined in a JSON config file,
handles parallelization, and saves results according to the data contract.
"""

import os
import json
import csv
import pathlib
import argparse
import uuid
from joblib import Parallel, delayed
from tqdm import tqdm
from .model import DOFTModel

def run_one_simulation(exp_config, run_params, out_dir):
    """
    Executes a single simulation run for a given configuration.
    """
    # Combine base config with run-specific parameters
    cfg = {**exp_config, **run_params}
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
    # 1. runs.csv entry
    run_row = {
        "run_id": run_id,
        "seed": cfg["seed"],
        "a_mean": cfg["a"]["mean"],
        "tau_mean": cfg["tau0"]["mean"],
        "gamma": cfg["gamma"],
        "ceff_pulse": run_result.get("ceff_pulse"),
        "ceff_pulse_ic95": None, # Placeholder for now
        "anisotropy_max_pct": run_result.get("anisotropy_max_pct"),
        "lpc_deltaK_neg_frac": run_result.get("lpc_deltaK_neg_frac"),
        "lpc_vcount": run_result.get("lpc_vcount")
    }

    # 2. blocks.csv entries
    for block in blocks_data:
        block["run_id"] = run_id

    # 3. run_meta.json
    meta_info = {
        "run_id": run_id,
        "config": cfg
        # In a real scenario, add git commit hash, versions, etc.
    }
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

    # --- Build the list of simulation tasks from the config ---
    tasks = []
    for experiment in config["experiments"]:
        base_cfg = {k: v for k, v in config.items() if k != "experiments"}
        base_cfg.update({k: v for k, v in experiment.items() if not isinstance(v, list)})
        
        # Expand parameter sweeps
        param_sweeps = {k: v for k, v in experiment.items() if isinstance(v, list)}
        if not param_sweeps:
             # Handle case with no sweeps, just seeds
            for seed in experiment.get("seeds", [0]):
                tasks.append((base_cfg, {"seed": seed}))
        else:
            # Generate all combinations of swept parameters
            import itertools
            keys, values = zip(*param_sweeps.items())
            for bundle in itertools.product(*values):
                run_params = dict(zip(keys, bundle))
                tasks.append((base_cfg, run_params))

    print(f"# Starting {len(tasks)} simulation runs with {args.n_jobs} parallel jobs...")

    # --- Run simulations in parallel ---
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_one_simulation)(base, params, out_dir) for base, params in tqdm(tasks)
    )

    # --- Aggregate and save results ---
    all_runs = [r[0] for r in results]
    all_blocks = [b for r in results for b in r[1]]

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
