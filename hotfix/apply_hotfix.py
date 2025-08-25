
#!/usr/bin/env python3
import sys, re, pathlib

root = pathlib.Path(__file__).resolve().parent.parent  # expect placed inside MODEL/hotfix/
model = root / "src" / "model.py"
if not model.exists():
    print(f"[hotfix] No se encontró {model}. ¿Estás ejecutando desde MODEL/?", file=sys.stderr)
    sys.exit(1)

code = model.read_text(encoding="utf-8")

# 1) Asegurar import torch
if re.search(r"^\s*import\s+torch\b", code, flags=re.M) is None:
    code = code.replace("import numpy as np", "import numpy as np\nimport torch")

# 2) Reemplazo seguro de la línea problemática np.sum(self.Y, axis=1)
pattern = r"\bMterm\s*=\s*np\.sum\(\s*self\.Y\s*,\s*axis\s*=\s*1\s*\)"
replacement = "Mterm = (torch.sum(self.Y, dim=1) if isinstance(self.Y, torch.Tensor) else np.sum(self.Y, axis=1))"
new_code, n = re.subn(pattern, replacement, code)
if n == 0:
    print("[hotfix] No encontré la línea 'Mterm = np.sum(self.Y, axis=1)'. ¿Ya estaba parcheado?", file=sys.stderr)
else:
    print(f"[hotfix] Reemplazos aplicados: {n}")

# 3) Guardar
model.write_text(new_code, encoding="utf-8")
print("[hotfix] OK: parche aplicado a src/model.py")
