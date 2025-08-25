
import os, json, csv, pathlib
from joblib import Parallel, delayed
from tqdm import tqdm
from .model import DOFTModel

def one(cfg, g, x, s, out_dir):
    m = DOFTModel(cfg)
    return m.run(gamma=float(g), xi_amp=float(x), seed=int(s), out_dir=out_dir)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="threading")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--log-interval", type=int, default=60)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg["log_interval"] = args.log_interval

    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    gammas = cfg.get("gammas", [0.0, 0.1, 0.3])
    xis = cfg.get("xis", [0.0, 1e-3, 3e-3])
    seeds = cfg.get("seeds", [0,1,2])
    combos = [(g,x,s) for g in gammas for x in xis for s in seeds]
    print(f"# backend:{args.backend} | tasks:{len(combos)} | USE_GPU={os.environ.get('USE_GPU','1')}")

    rows = Parallel(n_jobs=args.n_jobs, backend=args.backend, max_nbytes=None)(
        delayed(one)(cfg, g, x, s, str(out_dir)) for (g,x,s) in tqdm(combos)
    )

    table = out_dir / "table_full.csv"
    with open(table, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty:
        gsum = df.groupby("gamma").agg(
            ceff_bar=("ceff_bar","mean"),
            anisotropy_rel=("anisotropy_rel","mean"),
            hbar_eff=("hbar_eff","mean"),
            lpc_rate=("lpc_viol_frac","mean")
        ).reset_index()
        gsum.to_csv(out_dir/"summary.csv", index=False)
        print("# saved:", table, out_dir/"summary.csv")

if __name__ == "__main__":
    main()
