#!/usr/bin/env python3
"""Compare a fresh full run with the frozen acceptance metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = {
    "task1": {"f1": 0.6600541027953111, "precision": 0.7577639751552795, "recall": 0.5846645367412141},
    "task2": {"top1_caption_accuracy": 0.6771428571428572, "class_top1_accuracy": 0.8771428571428571},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("task1", "task2"))
    parser.add_argument("report", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.task == "task1":
        actual = report["metrics"]
    else:
        actual = {"top1_caption_accuracy": report["benchmark"]["task_metric"]["value"], **report["benchmark"]["extra_task_metrics"]}
    failed = False
    for key, expected in EXPECTED[args.task].items():
        value = float(actual[key])
        delta = value - expected
        print(f"{key}: actual={value:.6f} expected={expected:.6f} delta={delta:+.6f}")
        failed |= abs(delta) > args.tolerance
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
