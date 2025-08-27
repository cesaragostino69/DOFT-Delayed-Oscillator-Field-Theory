
import os, pandas as pd, matplotlib.pyplot as plt

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summ = os.path.join(args.indir, "summary.csv")
    if not os.path.exists(summ):
        print("No summary.csv at", summ)
        return
    df = pd.read_csv(summ)

    plt.figure(); plt.plot(df["gamma"], df["hbar_eff"], marker="o")
    plt.xlabel("gamma"); plt.ylabel("hbar_eff"); plt.title("ħ_eff vs gamma"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "hbar_eff_vs_gamma.png"), dpi=140); plt.close()

    plt.figure(); plt.plot(df["gamma"], df["lpc_rate"], marker="o")
    plt.xlabel("gamma"); plt.ylabel("LPC violation rate"); plt.title("LPC vs gamma"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "lpc_rate_vs_gamma.png"), dpi=140); plt.close()

    plt.figure(); plt.bar(df["gamma"].astype(str), df["anisotropy_rel"])
    plt.xlabel("gamma"); plt.ylabel("Δc/c"); plt.title("Average anisotropy"); plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "anisotropy_vs_gamma.png"), dpi=140); plt.close()

if __name__ == "__main__":
    main()
