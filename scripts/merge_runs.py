#!/usr/bin/env python3
import argparse, glob, os
import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required to merge runs. Please install pandas and try again.") from e

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="directories or shard patterns (e.g. runs/quick_shard*)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    # Locate all table.csv under each input
    files = []
    for pat in args.inputs:
        for root in glob.glob(pat):
            files.extend(glob.glob(os.path.join(root, "**", "table.csv"), recursive=True))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No table.csv found in the inputs. Did any run finish?")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[warn] could not read {f}: {e}")

    if not dfs:
        raise SystemExit("Could not read any valid CSV.")

    big = pd.concat(dfs, ignore_index=True)

    # Asegura tipos numéricos en lo que nos importa
    num_cols = ["gamma","xi","ceff_bar","beta_dyn","lpcV","samples","step","K","z0"]
    big = coerce_numeric(big, num_cols)

    # Derivados opcionales
    if "lpcV" in big.columns:
        big["lpc_ok"] = (big["lpcV"].fillna(0) == 0).astype(float)

    os.makedirs(args.out, exist_ok=True)
    big.to_csv(os.path.join(args.out, "merged.csv"), index=False)

    # Armado de agregados flexibles
    group = big.groupby(["gamma","xi"], as_index=False)
    agg_dict = {
        "ceff_bar": ["mean","std"],
    }
    if "beta_dyn" in big.columns:
        agg_dict["beta_dyn"] = ["mean","std"]
    if "lpc_ok" in big.columns:
        agg_dict["lpc_ok"] = ["mean"]

    summary = group.agg(agg_dict)
    # Aplana columnas MultiIndex
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    # Renombres amigables
    summary = summary.rename(columns={
        "ceff_bar_mean":"ceff_bar_mean",
        "ceff_bar_std":"ceff_bar_std",
        "beta_dyn_mean":"beta_dyn_mean",
        "beta_dyn_std":"beta_dyn_std",
        "lpc_ok_mean":"lpc_ok_frac"
    })

    # SEMs cuando hay std
    if "ceff_bar_std" in summary.columns:
        counts = group.size().reset_index(name="n")["n"]
        summary["n"] = counts
        summary["ceff_bar_sem"] = summary["ceff_bar_std"] / np.sqrt(summary["n"].clip(lower=1))
    else:
        summary["n"] = group.size().values

    summary = summary.sort_values(["gamma","xi"]).reset_index(drop=True)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    # Muestra breve
    pd.set_option("display.max_columns", None)
    print(f"[ok] merged={len(big)} filas  |  grupos={len(summary)}")
    print(summary.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
