commit 029d1daab31867ec6d1a3fa97005b50df6081014
Author: cesaragostino <90473152+cesaragostino@users.noreply.github.com>
Date:   Fri Aug 29 23:25:03 2025 -0300

    update config_phase1 run_sim.py
    
    Summary
    Added a --config argument with a default JSON path, letting simulations load sweep settings from a user-specified file instead of hardcoded values
    Replaced inline sweep definitions by parsing seeds, sweep groups, and numerical parameters from the JSON configuration file
    Stored the configuration source and parameters in run_meta.json so runs document their input setup for reproducibility

diff --git a/configs/config_phase1.json b/configs/config_phase1.json
index 66c6b9a..07147d5 100644
--- a/configs/config_phase1.json
+++ b/configs/config_phase1.json
@@ -1,46 +1,14 @@
 {
-  "grid_size": 128,
-  "gamma": 0.05,
-  "seeds": [101, 102, 103, 104, 105],
-  "sweep_groups": [
-    {
-      "name": "ExperimentA_Pulse_C_Emergence",
-      "experiment_type": "pulse",
-      "a": [
-        {"mean": 1.0, "std": 0.01},
-        {"mean": 1.2, "std": 0.01},
-        {"mean": 1.5, "std": 0.01},
-        {"mean": 1.0, "std": 0.01},
-        {"mean": 1.0, "std": 0.01}
-      ],
-      "tau0": [
-        {"mean": 1.0, "std": 0.01},
-        {"mean": 1.2, "std": 0.01},
-        {"mean": 1.5, "std": 0.01},
-        {"mean": 0.8, "std": 0.01},
-        {"mean": 0.67, "std": 0.01}
-      ]
-    }
-  ],
-  "numerical_params": {
-    "steps": 5000,
-    "dt": 0.002,
-    "dx": 1.0,
-    "log_interval": 10
-  },
-  "physics_params": {
-    "K": 0.3,
-    "xi_amp": 1e-4,
-    "prony_memory": {
-      "weights": [0.5, 0.3, 0.2],
-      "thetas": [0.01, 0.1, 1.0]
-    }
+  "seeds": [42, 123, 456, 789, 1011],
+  "sweep_groups": {
+    "g1": [[1.0, 1.0], [1.2, 1.2], [1.5, 1.5]],
+    "g2": [[1.0, 1.0], [1.2, 1.0], [1.5, 1.0]],
+    "g3": [[1.0, 1.0], [1.0, 0.8], [1.0, 0.67]]
   },
-  "metrics_params": {
-    "windowing": {
-      "len_steps": 2048,
-      "overlap": 0.5,
-      "detrend": "linear"
-    }
+  "numerical_params": {
+    "gamma": 0.05,
+    "grid_size": 100,
+    "a_ref": 1.0,
+    "tau_ref": 1.0
   }
 }
diff --git a/src/doft/simulation/run_sim.py b/src/doft/simulation/run_sim.py
index 35e4c61..8df5ad8 100644
--- a/src/doft/simulation/run_sim.py
+++ b/src/doft/simulation/run_sim.py
@@ -137,6 +137,15 @@ def main():
     This version uses the new IMEX/Leapfrog integrator with Prony memory.
     """
     parser = argparse.ArgumentParser(description="Run DOFT Phase-1 Simulation Sweep.")
+    default_config = os.environ.get(
+        "DOFT_CONFIG",
+        str(Path(__file__).resolve().parents[3] / "configs" / "config_phase1.json"),
+    )
+    parser.add_argument(
+        "--config",
+        default=default_config,
+        help="Path to JSON file with sweep configuration",
+    )
     parser.add_argument(
         "--boundary",
         choices=["periodic", "reflective", "absorbing"],
@@ -168,6 +177,7 @@ def main():
     )
     args = parser.parse_args()
 
+<<<<<<< ours
     env_cfg = _load_env_config()
     default_cfg = {
         "gamma": 0.05,
@@ -261,6 +271,27 @@ def main():
     a_ref = 1.0
     tau_ref = 1.0
 >>>>>>> theirs
+=======
+    # --- Load Configuration from JSON ---
+    with open(args.config, "r") as f:
+        loaded_cfg = json.load(f)
+
+    seeds = loaded_cfg.get("seeds", [])
+    sweep_groups = loaded_cfg.get("sweep_groups", {})
+    numerical_params = loaded_cfg.get("numerical_params", {})
+
+    gamma = numerical_params.get("gamma", 0.0)
+    grid_size = numerical_params.get("grid_size", 0)
+    a_ref = numerical_params.get("a_ref", 1.0)
+    tau_ref = numerical_params.get("tau_ref", 1.0)
+
+    simulation_points = []
+    point_to_group = {}
+    for gname, pts in sweep_groups.items():
+        for a_val, tau_val in pts:
+            simulation_points.append((a_val, tau_val))
+            point_to_group[(a_val, tau_val)] = gname
+>>>>>>> theirs
 
     # --- Create Unique Output Directory ---
     base_run_dir = 'runs'
@@ -344,6 +375,12 @@ def main():
         'total_runs_in_sweep': total_sims,
         'simulation_points': simulation_points,
         'seeds_used': seeds,
+        'config_source': args.config,
+        'config_params': {
+            'seeds': seeds,
+            'sweep_groups': sweep_groups,
+            'numerical_params': numerical_params,
+        },
         'fixed_params': {'gamma': gamma, 'grid_size': grid_size},
         'sweep_groups': sweep_groups,
         'config_file': args.config,
