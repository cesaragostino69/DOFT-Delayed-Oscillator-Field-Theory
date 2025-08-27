#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-runs/quick}"
PAR="${PAR:-4}"   # number of shards in parallel

for i in $(seq 0 $((PAR-1))); do
  OUT="${OUT_ROOT}_shard${i}"
  echo ">> shard $i  →  $OUT  (SEED_OFFSET=$((i*1000)))"
done

export SEED_OFFSET_BASE=0  # read internally by run_sim

# run shards in the background
pids=()
for i in $(seq 0 $((PAR-1))); do
  export SEED_OFFSET=$((i*1000))
  bash scripts/run_quick.sh "${OUT_ROOT}_shard${i}" &
  pids+=($!)
done

# wait for them to finish
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
exit $fail

