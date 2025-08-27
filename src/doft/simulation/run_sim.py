# src/doft/simulation/run_sim.py
import argparse
import pandas as pd
import numpy as np
import time
import json
import os
from itertools import product

# Import the physical model, which will now be complete
from doft.models.model import DOFTModel

def main():
    """
    Main orchestrator for the DOFT Phase 1 counter-trial.
    This script replaces the previous 'stub' and executes the experiments
    required by the "Definitive Action Plan".
    It saves the results of each complete sweep into a unique, timestamped directory.
    """
    parser = argparse.ArgumentParser(description="Run DOFT Phase-1 Simulation Sweep.")
    args = parser.parse_args()

    # --- "Break the Constant" Sweep Configuration (Experiment A) ---
    param_grid = {
        'group1': [(1.0, 1.0), (1.2, 1.2), (1.5, 1.5)], # constant ratio
        'group2': [(1.2, 1.0), (1.5, 1.0)],             # increase a
        'group3': [(1.0, 0.8), (1.0, 0.67)]             # decrease tau
    }
    base_point = [(1.0, 1.0)]
    simulation_points = base_point + param_grid['group1'][1:] + param_grid['group2'] + param_grid['group3']
    
    seeds = [42, 123, 456, 789, 1011] # ≥5 seeds per point, as required
    
    # Fixed parameters for Phase 1
    gamma = 0.05  # Requirement: γ > 0 for a passive system
    grid_size = 100 # Oscillator network size
    
    # --- Create Unique Output Directory ---
    # Create the main 'runs' directory if it doesn't exist
    base_run_dir = 'runs'
    os.makedirs(base_run_dir, exist_ok=True)
    
    # Generate the unique, timestamped directory for this specific execution
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_run_dir, f'phase1_run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Saving results to: {output_dir}")

    # --- Simulation Execution ---
    all_runs_data = []
    all_blocks_data = []
    
    print("🚀 Starting DOFT Phase-1 Simulation Sweep...")
    
    run_counter = 0
    total_sims = len(simulation_points) * len(seeds)

    for (a_mean, tau_mean) in simulation_points:
        for seed in seeds:
            run_counter += 1
            # The run_id is internal to the CSV files
            run_id = f"run_{int(time.time())}_{run_counter}"
            print(f"[{run_counter}/{total_sims}] Running sim: a={a_mean}, τ={tau_mean}, seed={seed}")
            
            # 1. Instantiate the model with the parameters for this run
            model = DOFTModel(
                grid_size=grid_size,
                a=a_mean,
                tau=tau_mean,
                gamma=gamma,
                seed=seed
            )
            
            # 2. Execute the simulation
            run_metrics, blocks_df = model.run()
            
            # 3. Enrich the results with the input parameters
            run_metrics['run_id'] = run_id
            run_metrics['seed'] = seed
            run_metrics['a_mean'] = a_mean
            run_metrics['tau_mean'] = tau_mean
            run_metrics['gamma'] = gamma
            all_runs_data.append(run_metrics)
            
            if blocks_df is not None and not blocks_df.empty:
                blocks_df['run_id'] = run_id
                all_blocks_data.append(blocks_df)

    # 4. Consolidate and save the results according to the data contract
    print(f"\n✅ Simulation sweep finished. Consolidating and writing results to {output_dir}...")

    # Save runs.csv
    runs_df = pd.DataFrame(all_runs_data)
    runs_output_path = os.path.join(output_dir, 'runs.csv')
    runs_df.to_csv(runs_output_path, index=False)
    print(f"--> Wrote {len(runs_df)} rows to {runs_output_path}")
    
    # Save blocks.csv
    if all_blocks_data:
        blocks_df_final = pd.concat(all_blocks_data, ignore_index=True)
        blocks_output_path = os.path.join(output_dir, 'blocks.csv')
        blocks_df_final.to_csv(blocks_output_path, index=False)
        print(f"--> Wrote {len(blocks_df_final)} rows to {blocks_output_path}")
    else:
        print("--> No block data generated for blocks.csv.")

    # Save metadata
    meta_data = {
        'run_directory': f'phase1_run_{timestamp}',
        'timestamp_utc': time.asctime(time.gmtime()),
        'total_runs_in_sweep': run_counter,
        'seeds_used': seeds,
        'simulation_points': simulation_points,
        'fixed_params': {'gamma': gamma, 'grid_size': grid_size},
        'analysis_params': {
            'lpc_window_size': 2048,
            'lpc_overlap': 0.5,
            'detrending': 'linear'
        }
    }
    meta_output_path = os.path.join(output_dir, 'run_meta.json')
    with open(meta_output_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
    print(f"--> Wrote metadata to {meta_output_path}")

if __name__ == "__main__":
    main()