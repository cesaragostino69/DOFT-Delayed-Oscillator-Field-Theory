# tests/test_analyze_results.py
import sys
from pathlib import Path

import pandas as pd


def test_analyze_results_main(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from doft.analysis.analyze_results import main

    indir = tmp_path / "in"
    indir.mkdir()
    df = pd.DataFrame({
        "gamma": [0.1, 0.2],
        "hbar_eff": [0.5, 0.6],
        "lpc_rate": [0.0, 0.1],
        "anisotropy_rel": [0.01, 0.02],
    })
    df.to_csv(indir / "summary.csv", index=False)

    outdir = tmp_path / "out"
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(sys, "argv", ["analyze_results", "--in", str(indir), "--out", str(outdir)])

    main()

    assert (outdir / "hbar_eff_vs_gamma.png").exists()
    assert (outdir / "lpc_rate_vs_gamma.png").exists()
    assert (outdir / "anisotropy_vs_gamma.png").exists()
    