#!/usr/bin/env bash
set -euo pipefail

PAR="${PAR:-4}"   # number of shards in parallel
export DOFT_CONFIG="${DOFT_CONFIG:-configs/config_phase1.json}"

echo "# using config: $DOFT_CONFIG"

for i in $(seq 0 $((PAR-1))); do
  echo ">> shard $i  (SEED_OFFSET=$((i*1000)))"
done

export SEED_OFFSET_BASE=0  # read internally by run_sim

# run shards in the background
pids=()
for i in $(seq 0 $((PAR-1))); do
  export SEED_OFFSET=$((i*1000))
  bash scripts/run_phase1.sh &
  pids+=($!)
done

# wait for them to finish
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
exit $fail

