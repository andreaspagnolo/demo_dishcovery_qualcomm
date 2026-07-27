#!/usr/bin/env python3
"""Verify the frozen input and model hashes before benchmarking."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    failures = 0
    for raw in (ROOT / "config/checksums/model_and_input_sha256.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.strip()
        actual = digest(path) if path.is_file() else "MISSING"
        state = "OK" if actual == expected else "FAIL"
        print(f"{state:4} {relative}")
        failures += state == "FAIL"
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
