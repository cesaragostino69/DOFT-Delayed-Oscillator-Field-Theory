#!/usr/bin/env python
"""Minimal simulation entry point for Phase 1.

This stub parses the configuration file and writes placeholder output so
that the run_phase1.sh script can execute during tests.
"""
import argparse
import json
import pathlib
import uuid
import csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dummy DOFT simulation")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--out", required=True, help="Directory to write results")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs (unused)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        config = json.load(f)

    runs_path = out_dir / "runs.csv"
    with open(runs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "seed"])
        writer.writeheader()
        for exp in config.get("experiments", []):
            for seed in exp.get("seeds", []):
                writer.writerow({"run_id": uuid.uuid4().hex, "seed": seed})

    print(f"Wrote placeholder results to {runs_path}")


if __name__ == "__main__":
    main()
