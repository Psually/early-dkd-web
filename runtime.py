"""Portable runtime; optional private training root via DKD_PROJECT_ROOT."""
from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DKD_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
PYDEPS = ROOT / "work" / "pydeps"
if PYDEPS.is_dir() and str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

def prepare_sklearn() -> None:
    if not PYDEPS.is_dir():
        import sklearn  # noqa: F401
        return
    name = "sklearn.__check_build._check_build"
    if name not in sys.modules:
        extension = next((PYDEPS / "sklearn" / "__check_build").glob("_check_build*.pyd"))
        spec = importlib.util.spec_from_file_location(name, extension)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load scikit-learn extension")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
