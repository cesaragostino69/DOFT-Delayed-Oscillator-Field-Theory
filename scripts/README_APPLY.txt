DOFT v12b - Patch Bundle (GPU and stability)

Contents:
- patches/patch_v12b.diff  -> unified diff (for git apply)
- scripts/apply_patch_v12b.py -> in-place patch (regex) if you don't use git or the diff doesn't apply
- scripts/merge_runs.py -> merges shards into a global summary
- scripts/doctor_cuda.py -> quick CUDA/Torch check

Option A (git):
  unzip doft_v12b_patch_bundle.zip -d .
  git checkout -b v12b_safety
  git apply patches/patch_v12b.diff
  pip install -r requirements.txt

Option B (without git, safer):
  unzip doft_v12b_patch_bundle.zip -d .
  python scripts/apply_patch_v12b.py
  pip install -r requirements.txt

Run shards in parallel:
  PAR=6 bash scripts/run_quick_multi.sh runs/quick
  python scripts/merge_runs.py runs/quick_shard* --out runs/quick_merged

Key changes:
- requirements.txt: add typing-extensions>=4.8
- utils.py: _to_numpy, ensure_numpy and spectral_entropy robust to short series
- model.py: support torch on GPU, consistent gamma_t, torch/numpy sum, add lpc_ok_frac
- run_sim.py: summary.csv grouped by (gamma, xi) with means and replica counts
