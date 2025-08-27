# tests/test_import.py
import importlib
import sys
from pathlib import Path


def test_import_doft():
    """Ensure the doft package imports without errors."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    importlib.import_module("doft")
