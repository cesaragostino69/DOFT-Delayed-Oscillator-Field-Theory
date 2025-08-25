#!/usr/bin/env python3
# Applies v12b changes in-place without git.
# - requirements.txt -> add typing-extensions>=4.8 if missing
# - src/utils.py -> add _to_numpy, ensure_numpy and robust spectral_entropy
# - src/model.py -> GPU/torch guard, gamma_t and numpy/torch sum; lpc_ok_frac
# - src/run_sim.py -> grouped summary by (gamma, xi)
# Creates .bak backups before modifying each file.

import re, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

def read(p): return p.read_text(encoding="utf-8")
def write(p, s):
    bak = p.with_suffix(p.suffix + ".bak")
    if p.exists():
        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(s, encoding="utf-8")
    print(f"patched: {p}  (backup: {bak.name})")

def patch_requirements():
    p = ROOT/"requirements.txt"
    if not p.exists():
        print("SKIP requirements.txt (missing)")
        return
    s = read(p)
    if "typing-extensions" not in s:
        s = s.rstrip() + "\ntyping-extensions>=4.8\n"
        write(p, s)
    else:
        print("OK requirements.txt (typing-extensions already present)")

def patch_utils():
    p = ROOT/"src"/"utils.py"
    if not p.exists():
        print("SKIP src/utils.py (missing)")
        return
    s = read(p)
    # insert _to_numpy after 'import numpy as np'
    if "_to_numpy(" not in s:
        s = re.sub(r"(import\s+numpy\s+as\s+np\s*\n)",
                   r"\1\n"
                   r"def _to_numpy(x):\n"
                   r"    try:\n"
                   r"        import torch\n"
                   r"        if isinstance(x, torch.Tensor):\n"
                   r"            return x.detach().float().cpu().numpy()\n"
                   r"    except Exception:\n"
                   r"        pass\n"
                   r"    import numpy as _np\n"
                   r"    return _np.asarray(x)\n\n",
                   s, count=1, flags=re.MULTILINE)
    # replace spectral_entropy body
    s = re.sub(
        r"def\s+spectral_entropy\([^\)]*\):\s*\n(?:.*\n)*?(?=^\S)",
        (
            "def spectral_entropy(x, eps=1e-12):\n"
            "    x = _to_numpy(x)\n"
            "    if x is None or x.size < 4:\n"
            "        return 0.0\n"
            "    import numpy as np\n"
            "    x = x - np.mean(x)\n"
            "    X = np.fft.rfft(x)\n"
            "    if X.size == 0:\n"
            "        return 0.0\n"
            "    P = (np.abs(X) ** 2)\n"
            "    Z = np.sum(P)\n"
            "    if Z <= eps:\n"
            "        return 0.0\n"
            "    P = P / Z\n"
            "    H = -np.sum(P * (np.log(P + eps)))\n"
            "    return float(H)\n\n"
        ),
        s, flags=re.MULTILINE
    )
    # add ensure_numpy if missing
    if "def ensure_numpy(" not in s:
        s += ("\n\ndef ensure_numpy(*arrs):\n"
              "    return tuple(_to_numpy(a) for a in arrs)\n")
    write(p, s)

def patch_model():
    p = ROOT/"src"/"model.py"
    if not p.exists():
        print("SKIP src/model.py (missing)")
        return
    s = read(p)
    # torch import guard
    if "_TORCH_OK" not in s:
        s = re.sub(r"(import\s+numpy\s+as\s+np[^\n]*\n)",
                   r"import os\n\1"
                   r"try:\n    import torch\n    _TORCH_OK = True\n"
                   r"except Exception:\n    torch = None\n    _TORCH_OK = False\n",
                   s, count=1)
    # __init__ device logic (best-effort inject)
    if "self.use_torch" not in s:
        s = re.sub(r"def\s+__init__\(\s*self,\s*cfg[^\)]*\):\s*\n\s*([^#\n].*\n)+?",
                   lambda m: m.group(0) +
                   "        use_gpu = bool(int(os.getenv('USE_GPU', '0')))\n"
                   "        if use_gpu:\n"
                   "            if not (_TORCH_OK and torch.cuda.is_available()):\n"
                   "                raise RuntimeError('GPU requested but torch is not available on this machine.')\n"
                   "            self.device = torch.device('cuda')\n"
                   "            self.dtype = torch.float32\n"
                   "            self.use_torch = True\n"
                   "        else:\n"
                   "            self.device = 'cpu'\n"
                   "            self.dtype = None\n"
                   "            self.use_torch = False\n",
                   s, count=1, flags=re.MULTILINE)
    # run(): set gamma_t and debug
    if "self.gamma_t" not in s:
        s = re.sub(r"(def\s+run\([^\)]*\):\s*\n)",
                   r"\1        if getattr(self, 'use_torch', False):\n"
                   r"            self.gamma_t = torch.tensor(float(gamma), device=self.device, dtype=self.dtype)\n"
                   r"            print(f"[debug] gamma_t={self.gamma_t.item():.6g} xi={xi_amp} device={self.device}")\n"
                   r"        else:\n"
                   r"            self.gamma_t = float(gamma)\n"
                   r"            print(f"[debug] gamma={self.gamma_t:.6g} xi={xi_amp} device=cpu")\n"
                   r"        self._step_counter = 0\n",
                   s, count=1, flags=re.MULTILINE)
    # step_euler: sum fix
    if "np.sum(self.Y, axis=1)" in s and "self.Y.sum(dim=1)" not in s:
        s = s.replace("np.sum(self.Y, axis=1)",
                      "(self.Y.sum(dim=1) if ('torch' in globals() and hasattr(self, 'use_torch') and self.use_torch and isinstance(self.Y, torch.Tensor)) else np.sum(self.Y, axis=1))")
    # add lpc_ok_frac next to lpc_viol_frac in row dict
    if "lpc_ok_frac" not in s and "lpc_viol_frac" in s:
        s = re.sub(r"(['\"]lpc_viol_frac['\"]\s*:\s*[^,]+,?\s*)\n",
                   r"\1\n            'lpc_ok_frac': 1.0 - float(lpc_viol_frac),\n",
                   s, count=1)
    write(p, s)

def patch_run_sim():
    p = ROOT/"src"/"run_sim.py"
    if not p.exists():
        print("SKIP src/run_sim.py (missing)")
        return
    s = read(p)
    # ensure groupby summary
    if "groupby([\"gamma\",\"xi\"]" not in s and "summary.csv" in s:
        s = re.sub(
            r"(df\.to_csv\([^\n]+table_full\.csv[^\n]+\)\s*\n)(.*summary.*\n.*\n)?",
            (
                "\1"
                "summary = (df.groupby([\"gamma\",\"xi\"], as_index=False)\n"
                "             .agg(ceff_bar=(\"ceff_bar\",\"mean\"),\n"
                "                  anisotropy_rel=(\"anisotropy_rel\",\"mean\"),\n"
                "                  hbar_eff=(\"hbar_eff\",\"mean\"),\n"
                "                  lpc_viol_frac=(\"lpc_viol_frac\",\"mean\"),\n"
                "                  lpc_ok_frac=(\"lpc_ok_frac\",\"mean\"),\n"
                "                  runs=(\"seed\",\"count\")))\n"
                "summary.to_csv(out_dir / \"summary.csv\", index=False)\n"
            ),
            s, count=1, flags=re.MULTILINE
        )
    write(p, s)

def main():
    patch_requirements()
    patch_utils()
    patch_model()
    patch_run_sim()
    print("\nListo. Ahora:\n"
          "  conda activate doft_v12 && pip install -r requirements.txt\n"
          "  USE_GPU=1 PAR=6 bash scripts/run_quick_multi.sh runs/quick_gpu\n"
          "y fusiona:\n"
          "  python scripts/merge_runs.py runs/quick_gpu_shard* --out runs/quick_gpu_merged\n")


if __name__ == "__main__":
    main()