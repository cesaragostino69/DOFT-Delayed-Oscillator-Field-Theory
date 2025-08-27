#!/usr/bin/env python
"""Run DOFT experiments described by a JSON configuration.

This utility reads ``config_phase1.json``-style files, instantiates a
``DOFTModel`` for every experiment/seed/gamma combination and records the
summary of each run to ``summary.csv`` inside the output directory.

The intent is not to be a high performance runner but to provide a simple
reference implementation that our documentation and tests can execute. Basic
parallel execution is available via the ``--n-jobs`` argument.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import multiprocessing as mp
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from doft.models.model import DOFTModel


def _ensure_list(x: Any) -> List[Any]:
    """Return ``x`` if it is a list, otherwise wrap it in a list."""
    if isinstance(x, list):
        return x
    return [x]


def _norm_params(params: Any) -> Dict[str, float]:
    """Normalise parameter dictionaries to ``{"mean": .., "sigma": ..}``.

    The configuration files may specify the dispersion either as ``std`` or
    ``sigma``. ``DOFTModel`` expects the latter.
    """
    if not isinstance(params, dict):
        return {"mean": float(params), "sigma": 0.0}
    if "sigma" in params:
        return {"mean": float(params["mean"]), "sigma": float(params["sigma"])}
    if "std" in params:
        return {"mean": float(params["mean"]), "sigma": float(params["std"])}
    return {"mean": float(params.get("mean", 0.0)), "sigma": 0.0}


def _run_single(task: Tuple[str, Dict[str, Any], float, str]) -> Dict[str, float]:
    """Execute a single simulation run.

    Parameters
    ----------
    task:
        Tuple containing the experiment name, configuration dictionary, noise
        amplitude and output directory path.

    Returns
    -------
    Dict[str, float]
        Row ready to be written to ``summary.csv``.
    """

    name, cfg, xi_amp, out_dir = task
    run_path = pathlib.Path(out_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    model = DOFTModel(cfg)
    results, _ = model.run(xi_amp=xi_amp, seed=int(cfg["seed"]), out_dir=str(run_path))

    return {
        "experiment": name,
        "gamma": float(cfg["gamma"]),
        "seed": int(cfg["seed"]),
        "ceff_pulse": results.get("ceff_pulse", 0.0),
        "ceff_x": results.get("ceff_x", 0.0),
        "ceff_y": results.get("ceff_y", 0.0),
        "anisotropy": results.get("anisotropy_max_pct", 0.0),
        "hbar_eff": results.get("hbar_eff", 0.0),
        "lpc_rate": results.get("lpc_rate", 0.0),
    }


def run_experiments(config: Dict[str, Any], out_dir: pathlib.Path, n_jobs: int = 1) -> None:
    """Execute all experiments defined in ``config``.

    A ``summary.csv`` file is produced with one row per executed run containing
    the main metrics returned by :meth:`DOFTModel.run` along with an estimate of
    ``hbar_eff`` computed from the final state.
    """
    base_cfg: Dict[str, Any] = {
        k: v
        for k, v in config.items()
        if k not in {"experiments", "prony_memory", "xi_amp"}
    }

    prony = config.get("prony_memory", {})
    base_cfg["prony_weights"] = prony.get("weights", [])
    base_cfg["prony_thetas"] = prony.get("thetas", [])

    xi_amp = float(config.get("xi_amp", 0.0))

    summary_path = out_dir / "summary.csv"
    fieldnames = [
        "experiment",
        "gamma",
        "seed",
        "ceff_pulse",
        "ceff_x",
        "ceff_y",
        "anisotropy",
        "hbar_eff",
        "lpc_rate",
    ]

    tasks: List[Tuple[str, Dict[str, Any], float, str]] = []

    for exp in config.get("experiments", []):
        name = exp.get("name", "experiment")
        seeds: Iterable[int] = _ensure_list(exp.get("seeds", [0]))
        gammas: Iterable[float] = _ensure_list(exp.get("gamma", 0.0))

        # Parameter lists for ``a`` and ``tau0``. If an experiment does not
        # specify them we fall back to the defaults in ``base_cfg`` (if any)
        # to keep the run configuration well defined.
        default_a = _norm_params(base_cfg.get("a", {"mean": 1.0, "sigma": 0.0}))
        default_tau = _norm_params(base_cfg.get("tau0", {"mean": 1.0, "sigma": 0.0}))

        a_params_list = [
            _norm_params(p) for p in _ensure_list(exp.get("a", [default_a]))
        ]
        tau_params_list = [
            _norm_params(p) for p in _ensure_list(exp.get("tau0", [default_tau]))
        ]

        for gamma in gammas:
            for idx, seed in enumerate(seeds):
                a_params = a_params_list[idx % len(a_params_list)]
                tau_params = tau_params_list[idx % len(tau_params_list)]

                cfg = dict(base_cfg)
                cfg.update(
                    {
                        "gamma": float(gamma),
                        "seed": int(seed),
                        "a": a_params,
                        "tau0": tau_params,
                    }
                )

                run_out = out_dir / name / f"g{gamma}_s{seed}"
                tasks.append((name, cfg, xi_amp, str(run_out)))

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if n_jobs == 1:
            for row in map(_run_single, tasks):
                writer.writerow(row)
        else:
            with mp.Pool(processes=n_jobs) as pool:
                for row in pool.imap_unordered(_run_single, tasks):
                    writer.writerow(row)

    print(f"Wrote summary results to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DOFT simulations")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--out", required=True, help="Directory to write results")
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel jobs"
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    run_experiments(config, out_dir, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
