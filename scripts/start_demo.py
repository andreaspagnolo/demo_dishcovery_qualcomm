#!/usr/bin/env python3
"""Start the frozen browser demo with its current defaults and no tuning surface."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DISHCOVERY_MODELS_DIR", str(ROOT / "external_assets/models"))
raise SystemExit(subprocess.call([sys.executable, str(ROOT / "demo_web/server.py")], cwd=ROOT))
