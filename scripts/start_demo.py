#!/usr/bin/env python3
"""Start the browser demo with the pinned local runtime defaults."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_ROOT = ROOT / ".venv_ort_qnn_245"
VENV_PYTHON = VENV_ROOT / "bin" / "python"
GENIEX_QNN_BACKEND = Path.home() / ".local/share/geniex/qairt/htp-files/libQnnHtp.so"


def ensure_demo_interpreter() -> None:
    """Re-exec in the pinned environment when launched through system Python."""
    if Path(sys.prefix).resolve() == VENV_ROOT.resolve():
        return
    if not VENV_PYTHON.is_file():
        raise SystemExit(
            f"Missing demo environment: {VENV_PYTHON}. "
            "Create it following README.md before starting the demo."
        )
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_demo_interpreter()

if not GENIEX_QNN_BACKEND.is_file():
    raise SystemExit(f"Missing GenieX QNN backend: {GENIEX_QNN_BACKEND}")

os.environ.setdefault("DISHCOVERY_MODELS_DIR", str(ROOT / "external_assets/models"))
os.environ.setdefault("GENIEX_TASK1_MODEL", "local/qwen3vl-4b-qairt-w4a16")
os.environ["GENIEX_QNN_BACKEND"] = str(GENIEX_QNN_BACKEND)
# ORT-QNN configures this path for its own QAIRT libraries. Inherited values
# can mix QAIRT copies and make the HTP backend fail during device creation.
os.environ.pop("ADSP_LIBRARY_PATH", None)
os.environ.pop("LD_LIBRARY_PATH", None)

try:
    exit_code = subprocess.call(
        [sys.executable, str(ROOT / "demo_web/server.py"), *sys.argv[1:]],
        cwd=ROOT,
    )
except KeyboardInterrupt:
    exit_code = 130
raise SystemExit(exit_code)
