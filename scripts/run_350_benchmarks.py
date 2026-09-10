#!/usr/bin/env python3
"""Run the two frozen 350-image benchmark configurations.

This intentionally exposes no tuning flags.  Change code only in a separate
experiment branch; preserve these commands as the acceptance baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
INPUTS = ROOT / "benchmark_inputs"
IMAGES = ROOT / "external_assets/images"
MODELS = Path(os.environ.get("DISHCOVERY_MODELS_DIR", ROOT / "external_assets/models"))
OUTPUT = ROOT / "run_outputs"
TASK1_GENIEX_MODEL = os.environ.get("GENIEX_TASK1_MODEL", "local/qwen3vl-4b-qairt-w8a16")


def required(value: str, name: str) -> str:
    if not value:
        raise SystemExit(f"{name} is required. See README.md, Environment and assets.")
    return value


def run(name: str, command: list[str]) -> None:
    run_dir = OUTPUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(
        json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "command": command}, indent=2) + "\n",
        encoding="utf-8",
    )
    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise SystemExit(f"{name} failed with status {completed.returncode}; inspect {log_path}")


def task1(python: str, qnn_backend: str) -> list[str]:
    return [
        python, str(PIPELINE / "task1_ingredients.py"),
        "--image-dir", str(IMAGES / "task1_350"),
        "--images-list", str(INPUTS / "task1_350/images.txt"),
        "--eval-samples", "350", "--eval-first", "--seed", "7",
        "--task1-logic", "legacy_siglip_qwen", "--legacy-keep-candidate-list",
        "--captions", str(INPUTS / "captions_cleaned.txt"),
        "--cleaned-json", str(INPUTS / "task1_ingredient_mapping.json"),
        "--image-ground-truth-map", str(INPUTS / "image_ground_truth_rows.csv"),
        "--siglip-backend", "onnx", "--siglip-context-mode", "split",
        "--siglip-image-size", "384", "--siglip-onnx-provider", "qnn", "--siglip-qnn-backend-path", qnn_backend,
        "--siglip-onnx-path", str(MODELS / "siglip2_qcs9075_out/fp16_powfix_split_qairt245/stage1/model.onnx"),
        "--siglip-onnx-stage2-path", str(MODELS / "siglip2_qcs9075_out/fp16_powfix_split_qairt245/stage2/model.onnx"),
        "--text-cache", str(INPUTS / "frozen_text_embeddings/orin_siglip2_text_feats_cache.npz"),
        "--vlm-model", TASK1_GENIEX_MODEL, "--vlm-family", "qwen", "--vlm-backend", "geniex",
        "--geniex-image-max-side", "512", "--geniex-image-min-side", "512", "--geniex-padding-mode", "white",
        "--measure", "--output-dir", str(OUTPUT / "task1_350"),
        "--output-json", str(OUTPUT / "task1_350/eval_first_350_samples.json"),
        "--predictions-csv", str(OUTPUT / "task1_350/eval_first_350_samples_predictions.csv"),
    ]


def task2(python: str, qnn_backend: str) -> list[str]:
    return [
        python, str(PIPELINE / "task2_caption_retrieval.py"),
        "--image-dir", str(IMAGES / "task2_350"),
        "--images-list", str(INPUTS / "task2_350/images.txt"),
        "--manifest", str(INPUTS / "task2_350/manifest.csv"),
        "--evaluation-json", str(INPUTS / "evaluation_data.json"),
        "--eval-samples", "350", "--eval-first", "--seed", "42", "--measure", "--strict-evk-stack",
        "--caption-text-mode", "class_caption", "--siglip-backend", "onnx", "--siglip-context-mode", "split",
        "--siglip-image-size", "384", "--siglip-top-k", "5", "--siglip-onnx-provider", "qnn",
        "--siglip-onnx-path", str(MODELS / "siglip2_qcs9075_out/fp16_powfix_split_qairt245/stage1/model.onnx"),
        "--siglip-onnx-stage2-path", str(MODELS / "siglip2_qcs9075_out/fp16_powfix_split_qairt245/stage2/model.onnx"),
        "--siglip-qnn-backend-path", qnn_backend,
        "--reranker-backend", "mtmd",
        "--reranker-gguf", str(MODELS / "Qwen3-VL-Reranker-2B-GGUF/Qwen3-VL-Reranker-2B-Q8_0-F16Emb-F16Cls-HTP.gguf"),
        "--reranker-mmproj", str(MODELS / "Qwen3-VL-Reranker-2B-GGUF/mmproj-Qwen3-VL-Reranker-2B-F16.gguf"),
        "--reranker-mtmd-executable", str(ROOT / "model_build/qwen3_vl_reranker/build/qwen3-vl-reranker-mtmd"),
        "--reranker-mtmd-compute", "hybrid", "--reranker-mtmd-threads", "8",
        "--reranker-context-size", "1024", "--reranker-batch-capacity", "512",
        "--reranker-image-max-tokens", "512", "--reranker-flash-attention", "off", "--reranker-individual-calls",
        "--rerank-top-k", "5", "--final-score-mode", "siglip_guarded", "--siglip-keep-gap", "3.0",
        "--checkpoint-json", str(OUTPUT / "task2_350/checkpoint.json"),
        "--output-dir", str(OUTPUT / "task2_350"),
        "--output-json", str(OUTPUT / "task2_350/eval_first_350_caption_alignment.json"),
        "--predictions-csv", str(OUTPUT / "task2_350/eval_first_350_caption_alignment.csv"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("task1", "task2", "all"))
    parser.add_argument("--task1-python", default=os.environ.get("TASK1_PYTHON", sys.executable))
    parser.add_argument("--task2-python", default=os.environ.get("TASK2_PYTHON", sys.executable))
    parser.add_argument("--qnn-backend-path", default=os.environ.get("GENIEX_QNN_BACKEND", ""))
    args = parser.parse_args()
    if args.task in {"task1", "all"}:
        run("task1_350", task1(args.task1_python, required(args.qnn_backend_path, "--qnn-backend-path")))
    if args.task in {"task2", "all"}:
        run("task2_350", task2(args.task2_python, required(args.qnn_backend_path, "--qnn-backend-path")))


if __name__ == "__main__":
    main()
