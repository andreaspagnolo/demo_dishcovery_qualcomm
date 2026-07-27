#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image, ImageFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evk_acceleration import (
    FIXED_IMAGE_SIZE,
    RERANKER_COMPONENTS_MS,
    SIGLIP_COMPONENTS,
    artifact_record,
    atomic_write_json,
    base_run_config,
    component_summary,
    load_and_check_evk_stack,
    process_memory_snapshot,
    recommended_context_capacity,
)
from measurement_utils import PowerMonitor, build_benchmark_summary, print_benchmark_summary, query_nvpmodel
from model_build.qwen3_vl_reranker import MtmdWorkerError, QairtWorkerError


# Resolve the project root robustly. If this file is inside a scripts/ folder,
# use the parent project folder; if it is already in the project root, keep it.
_THIS_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
MODEL_ROOT = Path(os.environ.get("DISHCOVERY_MODELS_DIR", ROOT / "external_assets/models"))
DEFAULT_SUBSET_ROOT = ROOT / "benchmark_inputs/task2_350"
DEFAULT_IMAGE_DIR = ROOT / "external_assets/images/task2_350"
DEFAULT_MANIFEST = DEFAULT_SUBSET_ROOT / "manifest.csv"
DEFAULT_IMAGES_LIST = DEFAULT_SUBSET_ROOT / "images.txt"
DEFAULT_EVALUATION_JSON = ROOT / "benchmark_inputs/evaluation_data.json"
DEFAULT_OUTPUT_DIR = ROOT / "run_outputs/task2_350"
DEFAULT_SIGLIP_MODEL = "hf-hub:timm/ViT-gopt-16-SigLIP2-384"
DEFAULT_SIGLIP_PRETRAINED = ""
DEFAULT_SIGLIP_BACKEND = "onnx"
DEFAULT_SIGLIP_ONNX_STAGE1 = MODEL_ROOT / "siglip2_qcs9075_out/fp16_powfix_split_qairt245/stage1/model.onnx"
DEFAULT_SIGLIP_ONNX_STAGE2 = MODEL_ROOT / "siglip2_qcs9075_out/fp16_powfix_split_qairt245/stage2/model.onnx"
DEFAULT_RERANKER_SIZE = "2B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_MTMD_ROOT = ROOT / "model_build/qwen3_vl_reranker"
DEFAULT_QAIRT_RERANKER_ROOT = ROOT / "model_build/qwen3_vl_reranker"
DEFAULT_QAIRT_RERANKER_EXECUTABLE = DEFAULT_QAIRT_RERANKER_ROOT / "build/qwen3-vl-reranker-qairt"
DEFAULT_QAIRT_RERANKER_CONTRACT = ROOT / "config/model_artifacts/qwen3_vl_reranker_2b_qairt_contract.json"
DEFAULT_QAIRT_QNN_BACKEND = Path(
    os.environ.get("GENIEX_QNN_BACKEND", "/home/ubuntu/.local/share/geniex/qairt/htp-files/libQnnHtp.so")
)
DEFAULT_MTMD_MODEL_DIR = MODEL_ROOT / "Qwen3-VL-Reranker-2B-GGUF"
DEFAULT_MTMD_MODEL = DEFAULT_MTMD_MODEL_DIR / "Qwen3-VL-Reranker-2B-Q8_0-F16Emb-F16Cls-HTP.gguf"
DEFAULT_MTMD_MMPROJ = DEFAULT_MTMD_MODEL_DIR / "mmproj-Qwen3-VL-Reranker-2B-F16.gguf"
RERANKER_MODEL_BY_SIZE = {
    "2B": DEFAULT_RERANKER_MODEL,
    "8B": "Qwen/Qwen3-VL-Reranker-8B",
}
RERANKER_REQUIRED_FILES = ("config.json", "modules.json", "tokenizer.json")
CAPTION_CACHE_VERSION = "orin_task2_siglip2_caption_bank_v1"
IMAGE_CACHE_VERSION = "orin_task2_siglip2_image_bank_v1"

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class Food500Row:
    cat: str
    filename: str
    caption: str


@dataclass(frozen=True)
class CaptionBankItem:
    caption_id: int
    cat: str
    filename: str
    caption: str
    text: str


@dataclass
class CaptionCandidate:
    caption_id: int
    cat: str
    filename: str
    caption: str
    text: str
    siglip_score: float
    rerank_score: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task 2 caption-alignment pipeline: SigLIP2 recalls candidate captions, "
            "then a Qwen3-VL reranker scores image-caption pairs."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Run one image. Accepts a full path or a path relative to --image-dir.",
    )
    parser.add_argument("--eval-samples", type=int, default=0, help="Evaluate N images.")
    parser.add_argument("--eval-all", action="store_true", help="Evaluate every image found in --evaluation-json.")
    parser.add_argument("--eval-first", action="store_true", help="Use the first N rows instead of a seeded random sample.")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Run images without ground truth and skip accuracy metrics. "
            "Useful for Dishcovery Test2 with a flat captions.json bank."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--images-list",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited image list, relative to --image-dir. "
            f"For the 350-image benchmark use {DEFAULT_IMAGES_LIST}."
        ),
    )
    parser.add_argument("--evaluation-json", type=Path, default=DEFAULT_EVALUATION_JSON)
    parser.add_argument(
        "--caption-bank-json",
        type=Path,
        default=None,
        help=(
            "JSON containing candidate captions. Defaults to --evaluation-json. "
            "Each row must have cat, filename, caption."
        ),
    )
    parser.add_argument(
        "--caption-text-mode",
        choices=("caption", "class_caption"),
        default="class_caption",
        help="Text passed to SigLIP/Qwen. class_caption prefixes the food class before the caption.",
    )
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--siglip-pretrained", default=DEFAULT_SIGLIP_PRETRAINED)
    parser.add_argument(
        "--siglip-backend",
        choices=("onnx", "open_clip"),
        default=DEFAULT_SIGLIP_BACKEND,
        help="EVK default is the split FP16 visual encoder through QNN; open_clip is the original path.",
    )
    parser.add_argument(
        "--siglip-context-mode",
        choices=("split", "single"),
        default="split",
        help="Use two established QNN contexts or one fused visual-encoder context.",
    )
    parser.add_argument("--siglip-onnx-path", type=Path, default=DEFAULT_SIGLIP_ONNX_STAGE1)
    parser.add_argument("--siglip-onnx-stage2-path", type=Path, default=DEFAULT_SIGLIP_ONNX_STAGE2)
    parser.add_argument("--siglip-onnx-provider", choices=("qnn", "gpu", "cpu", "auto"), default="qnn")
    parser.add_argument(
        "--siglip-qnn-backend-path",
        type=Path,
        default=None,
        help="Explicit libQnnHtp.so; point to GenieX's backend to keep one QAIRT stack.",
    )
    parser.add_argument("--siglip-onnx-input", default="")
    parser.add_argument("--siglip-onnx-output", default="")
    parser.add_argument("--siglip-io-binding", action="store_true")
    parser.add_argument(
        "--siglip-qnn-shared-memory-allocator",
        action="store_true",
        help="Opt in to ORT QNN's HTP shared-memory allocator after stack validation.",
    )
    parser.add_argument(
        "--siglip-qnn-finalization-mode",
        choices=("default", "0", "1", "2", "3"),
        default="default",
        help="QNN HTP graph-finalization optimization mode; profile memory and load time.",
    )
    parser.add_argument("--siglip-image-mean", default="0.5,0.5,0.5")
    parser.add_argument("--siglip-image-std", default="0.5,0.5,0.5")
    parser.add_argument("--siglip-image-size", type=int, default=384)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--torch-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument(
        "--siglip-top-k",
        type=int,
        default=5,
        help="Caption recall depth before Qwen reranking.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=5,
        help="Number of caption candidates to send to Qwen per image.",
    )
    parser.add_argument(
        "--final-score-mode",
        choices=("rerank", "siglip", "blend_z", "siglip_guarded"),
        default="siglip_guarded",
        help=(
            "How to choose the final caption. siglip_guarded keeps SigLIP top-1 "
            "when it is confident and uses Qwen only for ambiguous images."
        ),
    )
    parser.add_argument("--rerank-weight", type=float, default=0.7, help="Qwen weight for --final-score-mode blend_z.")
    parser.add_argument(
        "--siglip-keep-gap",
        type=float,
        default=3.0,
        help=(
            "For --final-score-mode siglip_guarded: keep SigLIP top-1 without Qwen "
            "when its score gap over rank 2 is at least this value."
        ),
    )
    parser.add_argument(
        "--reranker-size",
        default=DEFAULT_RERANKER_SIZE,
        help=(
            "Qwen3-VL reranker size alias, for example 8B or 2B. "
            "Unknown values are expanded as Qwen/Qwen3-VL-Reranker-<size>. "
            "Ignored when --reranker-model is provided."
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help=(
            "Explicit reranker repo ID or local path. Overrides --reranker-size. "
            f"Default is {DEFAULT_RERANKER_MODEL}."
        ),
    )
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--reranker-attn-implementation", default="sdpa")
    parser.add_argument("--reranker-local-files-only", action="store_true")
    parser.add_argument(
        "--reranker-backend",
        choices=("transformers", "mtmd", "qairt"),
        default="mtmd",
        help="Use Transformers, the Qualcomm GGUF/libmtmd bridge, or the native QAIRT scorer.",
    )
    parser.add_argument(
        "--reranker-gguf",
        type=Path,
        default=DEFAULT_MTMD_MODEL,
        help="Quantized rank-pooling GGUF used by --reranker-backend mtmd.",
    )
    parser.add_argument(
        "--reranker-mmproj",
        type=Path,
        default=DEFAULT_MTMD_MMPROJ,
        help="Qwen3-VL multimodal projector GGUF used by the mtmd backend.",
    )
    parser.add_argument(
        "--reranker-mtmd-executable",
        type=Path,
        default=DEFAULT_MTMD_ROOT / "build/qwen3-vl-reranker-mtmd",
    )
    parser.add_argument("--reranker-mtmd-compute", choices=("npu", "hybrid", "cpu"), default="hybrid")
    parser.add_argument("--reranker-mtmd-threads", type=int, default=8)
    parser.add_argument("--reranker-context-size", type=int, default=4096)
    parser.add_argument("--reranker-batch-capacity", type=int, default=512)
    parser.add_argument(
        "--reranker-flash-attention",
        choices=("off", "on", "auto"),
        default="off",
        help="Enable only after the backend reports support and ranking equivalence is verified.",
    )
    parser.add_argument(
        "--reranker-mtmd-max-batch-candidates",
        type=int,
        default=4,
        help="Maximum same-image candidates scored in one mtmd call. Use 1 to disable candidate batching.",
    )
    parser.add_argument("--reranker-image-max-tokens", type=int, default=512)
    parser.add_argument("--reranker-startup-timeout-sec", type=float, default=300.0)
    parser.add_argument("--reranker-request-timeout-sec", type=float, default=60.0)
    parser.add_argument(
        "--no-reranker-restart",
        action="store_true",
        help="Do not restart the reranker worker after a pipe failure, exit, or request timeout.",
    )
    parser.add_argument(
        "--reranker-qairt-executable",
        type=Path,
        default=DEFAULT_QAIRT_RERANKER_EXECUTABLE,
        help="Native JSON-lines QAIRT worker used by --reranker-backend qairt.",
    )
    parser.add_argument(
        "--reranker-qairt-bundle",
        type=Path,
        default=None,
        help="Compiled QAIRT reranker bundle directory or archive. Required by the qairt backend.",
    )
    parser.add_argument(
        "--reranker-qairt-contract",
        type=Path,
        default=DEFAULT_QAIRT_RERANKER_CONTRACT,
        help="Native reranker graph and precision contract passed to the QAIRT worker.",
    )
    parser.add_argument(
        "--reranker-qairt-backend-path",
        type=Path,
        default=DEFAULT_QAIRT_QNN_BACKEND,
        help="Pinned QAIRT HTP backend used by the native reranker worker.",
    )
    parser.add_argument("--reranker-qairt-verbose", action="store_true")
    parser.add_argument(
        "--reranker-individual-calls",
        action="store_true",
        help="Score each candidate in its own IPC request while reusing the resident image-KV prefix.",
    )
    parser.add_argument("--reranker-mtmd-verbose", action="store_true")
    parser.add_argument("--detailed-profiling", action="store_true")
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=None,
        help="Optional file containing a Hugging Face token.",
    )
    parser.add_argument(
        "--mock-reranker",
        action="store_true",
        help="Do not load Qwen; use SigLIP scores as rerank scores. Useful for smoke tests.",
    )
    parser.add_argument(
        "--caption-cache",
        type=Path,
        default=ROOT / "benchmark_inputs/frozen_text_embeddings/orin_task2_siglip2_caption_bank_cache.npz",
    )
    parser.add_argument(
        "--image-cache",
        type=Path,
        default=ROOT / "run_outputs/task2_350/image_cache.npz",
    )
    parser.add_argument("--no-caption-cache", action="store_true")
    parser.add_argument("--no-image-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-json",
        type=Path,
        default=None,
        help="Atomically save every completed image and native worker failure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume successful rows from --checkpoint-json; resumed runs are quality-only.",
    )
    parser.add_argument(
        "--evk-stack-manifest",
        type=Path,
        default=ROOT / "config/platform/evk_qairt_stack.json",
        help="Expected ORT-QNN/GenieX QAIRT compatibility contract.",
    )
    parser.add_argument(
        "--strict-evk-stack",
        action="store_true",
        help="Stop before model loading if the QAIRT stack contract is not satisfied.",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Use per-query evaluation and add latency, throughput, power, and energy summary.",
    )
    parser.add_argument("--measure-power", action="store_true", help="Sample Jetson tegrastats power during the per-query inference loop.")
    parser.add_argument("--power-sample-interval-ms", type=int, default=200)
    parser.add_argument("--w-config", default="", help="Power-budget label for the run, for example 15W, 30W, or 50W.")
    args = parser.parse_args()
    if args.reranker_mtmd_max_batch_candidates <= 0:
        parser.error("--reranker-mtmd-max-batch-candidates must be positive")
    if args.reranker_context_size <= 0 or args.reranker_batch_capacity <= 0:
        parser.error("--reranker-context-size and --reranker-batch-capacity must be positive")
    if args.reranker_startup_timeout_sec <= 0 or args.reranker_request_timeout_sec <= 0:
        parser.error("reranker startup/request timeouts must be positive")
    if args.siglip_image_size != FIXED_IMAGE_SIZE:
        parser.error("SigLIP2 and the reranker visual path are fixed at 384x384")
    if args.resume and args.checkpoint_json is None:
        parser.error("--resume requires --checkpoint-json")
    resolve_reranker_model_args(args)
    return args


def normalize_reranker_size(value: str) -> str:
    text = str(value or DEFAULT_RERANKER_SIZE).strip()
    if not text:
        text = DEFAULT_RERANKER_SIZE
    return text.upper()


def reranker_model_from_size(size: str) -> str:
    normalized = normalize_reranker_size(size)
    return RERANKER_MODEL_BY_SIZE.get(normalized, f"Qwen/Qwen3-VL-Reranker-{normalized}")


def resolve_reranker_model_args(args: argparse.Namespace) -> None:
    args.reranker_size = normalize_reranker_size(args.reranker_size)
    if args.reranker_model:
        args.reranker_model = str(args.reranker_model).strip()
        return
    args.reranker_model = reranker_model_from_size(args.reranker_size)


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def canonical_filename(value: str) -> str:
    """Canonical comparison key without changing cache/report spelling."""
    return unicodedata.normalize("NFC", normalize_text(value))


def display_name(class_name: str) -> str:
    return normalize_text(class_name.replace("_", " "))


def make_caption_text(row: Food500Row, mode: str) -> str:
    caption = normalize_text(row.caption)
    if not row.cat:
        return caption
    if mode == "caption":
        return caption
    if mode == "class_caption":
        return normalize_text(f"{display_name(row.cat)}. {caption}")
    raise ValueError(f"Unsupported caption-text mode: {mode}")


def sha256_jsonable(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_food500_rows(path: Path) -> list[Food500Row]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"{path} must contain a list")
    rows: list[Food500Row] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"{path} row {idx} must be an object")
        cat = normalize_text(item.get("cat", ""))
        filename = normalize_text(item.get("filename", ""))
        caption = normalize_text(item.get("caption", ""))
        if not cat or not filename or not caption:
            raise ValueError(f"{path} row {idx} is missing cat, filename, or caption")
        rows.append(Food500Row(cat=cat, filename=filename, caption=caption))
    return rows


def read_manifest(path: Path) -> list[Food500Row]:
    if not path.exists():
        return []
    rows: list[Food500Row] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"cat", "filename"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for item in reader:
            rows.append(
                Food500Row(
                    cat=normalize_text(item.get("cat", "")),
                    filename=normalize_text(item.get("filename", "")),
                    caption=normalize_text(item.get("caption", "")),
                )
            )
    return rows


def read_image_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Image list does not exist: {path}")
    names = [normalize_text(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Image list is empty: {path}")
    keys = [canonical_filename(name) for name in names]
    duplicates = sorted({name for name in keys if keys.count(name) > 1})
    if duplicates:
        raise ValueError(f"Image list contains duplicate entries: {duplicates[:10]}")
    return names


def resolve_image_list_names(image_dir: Path, names: list[str]) -> tuple[list[str], list[str]]:
    missing = [name for name in names if not (image_dir / name).is_file()]
    if not missing:
        return names, []
    normalized_paths: dict[str, list[str]] = {}
    normalized_basenames: dict[str, list[str]] = {}
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        relative_name = path.relative_to(image_dir).as_posix()
        key = canonical_filename(relative_name)
        normalized_paths.setdefault(key, []).append(relative_name)
        normalized_basenames.setdefault(canonical_filename(path.name), []).append(relative_name)
    resolved: list[str] = []
    unresolved: list[str] = []
    for name in names:
        if (image_dir / name).is_file():
            resolved.append(name)
            continue
        candidates = normalized_paths.get(canonical_filename(name), [])
        if not candidates:
            candidates = normalized_basenames.get(canonical_filename(Path(name).name), [])
        if len(candidates) == 1:
            replacement = candidates[0]
            print(f"Resolved Unicode-normalized image path: {name!r} -> {replacement!r}", flush=True)
            resolved.append(replacement)
        else:
            unresolved.append(name)
    return resolved, unresolved


def discover_images(image_dir: Path, gt_by_filename: dict[str, Food500Row]) -> list[Food500Row]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    rows: list[Food500Row] = []
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel = path.relative_to(image_dir).as_posix()
        gt = gt_by_filename.get(canonical_filename(rel))
        if gt is not None:
            rows.append(gt)
    return rows


def resolve_image_name(image: Path, image_dir: Path) -> tuple[str, Path]:
    if image.exists():
        image_path = image
        try:
            image_name = image_path.relative_to(image_dir).as_posix()
        except ValueError:
            image_name = image_path.name
        return image_name, image_path
    candidate = image_dir / image
    if candidate.exists():
        return image.as_posix(), candidate
    candidate = image_dir / image.name
    if candidate.exists():
        return image.name, candidate
    raise FileNotFoundError(f"Image does not exist: {image}")


def select_eval_rows(
    args: argparse.Namespace,
    evaluation_rows: list[Food500Row],
    manifest_rows: list[Food500Row],
) -> list[Food500Row]:
    gt_by_filename = {canonical_filename(row.filename): row for row in evaluation_rows}
    if args.images_list is not None:
        names = read_image_list(args.images_list)
        missing_gt = [name for name in names if canonical_filename(name) not in gt_by_filename]
        if missing_gt:
            raise FileNotFoundError(f"{len(missing_gt)} image-list entries are missing from {args.evaluation_json}: {missing_gt[:10]}")
        rows = [gt_by_filename[canonical_filename(name)] for name in names]
    elif manifest_rows:
        rows = [
            gt_by_filename[canonical_filename(row.filename)]
            for row in manifest_rows
            if canonical_filename(row.filename) in gt_by_filename
        ]
    else:
        rows = evaluation_rows
    if args.eval_all:
        return rows
    if args.eval_samples <= 0:
        raise SystemExit("Provide --image, --eval-samples N, or --eval-all.")
    if args.eval_samples >= len(rows):
        return rows
    if args.eval_first:
        return rows[: args.eval_samples]
    rng = random.Random(args.seed)
    selected = rng.sample(rows, args.eval_samples)
    return sorted(selected, key=lambda row: row.filename)


def build_caption_bank(rows: list[Food500Row], caption_text_mode: str) -> list[CaptionBankItem]:
    bank: list[CaptionBankItem] = []
    for idx, row in enumerate(rows):
        bank.append(
            CaptionBankItem(
                caption_id=idx,
                cat=row.cat,
                filename=row.filename,
                caption=row.caption,
                text=make_caption_text(row, caption_text_mode),
            )
        )
    if not bank:
        raise ValueError("No candidate captions were built")
    return bank


def read_caption_bank_items(path: Path, caption_text_mode: str) -> list[CaptionBankItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"{path} must contain a list")
    if not raw:
        raise ValueError(f"{path} is empty")
    if all(isinstance(item, str) for item in raw):
        bank: list[CaptionBankItem] = []
        for idx, item in enumerate(raw):
            caption = normalize_text(item)
            if not caption:
                continue
            bank.append(
                CaptionBankItem(
                    caption_id=idx,
                    cat="",
                    filename=f"caption_{idx:06d}",
                    caption=caption,
                    text=caption,
                )
            )
        if not bank:
            raise ValueError(f"{path} does not contain usable captions")
        return bank
    return build_caption_bank(read_food500_rows(path), caption_text_mode)


def select_inference_rows(args: argparse.Namespace) -> list[Food500Row]:
    if args.images_list is not None:
        names = read_image_list(args.images_list)
    else:
        suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        names = sorted(path.name for path in args.image_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
    names, missing = resolve_image_list_names(args.image_dir, names)
    if missing:
        raise FileNotFoundError(f"{len(missing)} image-list entries are missing from {args.image_dir}: {missing[:10]}")
    rows = [Food500Row(cat="", filename=name, caption="") for name in names]
    if args.eval_all:
        return rows
    if args.eval_samples <= 0:
        raise SystemExit("Provide --eval-samples N or --eval-all with --inference-only.")
    if args.eval_samples >= len(rows):
        return rows
    if args.eval_first:
        return rows[: args.eval_samples]
    rng = random.Random(args.seed)
    return sorted(rng.sample(rows, args.eval_samples), key=lambda row: row.filename)


def hf_hub_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def snapshot_model_weights_complete(snapshot: Path) -> bool:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = snapshot / name
        if path.is_file():
            return True

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.exists():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            return False
        shard_names = {str(name) for name in weight_map.values() if str(name)}
        return bool(shard_names) and all((snapshot / shard_name).is_file() for shard_name in shard_names)

    return False


def cached_hf_snapshot(repo_id: str, required_files: tuple[str, ...] = (), require_model_weights: bool = False) -> Path | None:
    cache_dir = hf_hub_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    candidates: list[Path] = []
    ref_path = cache_dir / "refs" / "main"
    if ref_path.exists():
        try:
            revision = ref_path.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(snapshots_dir / revision)
        except OSError:
            pass
    try:
        candidates.extend(sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True))
    except OSError:
        pass
    seen: set[Path] = set()
    for snapshot in candidates:
        if snapshot in seen:
            continue
        seen.add(snapshot)
        if (
            snapshot.is_dir()
            and all((snapshot / name).exists() for name in required_files)
            and (not require_model_weights or snapshot_model_weights_complete(snapshot))
        ):
            return snapshot
    return None


def extract_env_value(line: str, key: str) -> str | None:
    raw = line.strip()
    if not raw or raw.startswith("#") or "=" not in raw:
        return None
    name, value = raw.split("=", 1)
    if name.strip() != key:
        return None
    value = value.strip().strip("'").strip('"')
    return value or None


def read_token_file(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        for line in text.splitlines():
            value = extract_env_value(line, key)
            if value:
                return value
    return text.splitlines()[0].strip().strip("'").strip('"') or None


def resolve_hf_token(args: argparse.Namespace) -> str | None:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    token_paths: list[Path] = []
    if args.hf_token_file is not None:
        token_paths.append(args.hf_token_file.expanduser())
    token_paths.extend([ROOT / ".env", ROOT / ".hf_token", Path.home() / ".cache/huggingface/token"])
    for path in token_paths:
        token = read_token_file(path)
        if token:
            return token
    return None


def configure_hf_token(args: argparse.Namespace) -> str | None:
    token = resolve_hf_token(args)
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        print("Using Hugging Face token from environment/file.")
    else:
        print("No Hugging Face token found; downloads will be unauthenticated.")
    return token


def resolve_open_clip_source(model_name: str) -> str:
    if not model_name.startswith("hf-hub:"):
        return model_name
    repo_id = model_name.removeprefix("hf-hub:")
    snapshot = cached_hf_snapshot(repo_id, ("open_clip_config.json", "open_clip_model.safetensors", "tokenizer.json"))
    if snapshot is None:
        return model_name
    return f"local-dir:{snapshot}"


def require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "This pipeline requires PyTorch. On Jetson Orin, use an NVIDIA/JetPack-compatible "
            "PyTorch build, then install open_clip_torch, transformers, sentence-transformers, and pillow."
        ) from exc
    return torch


def torch_tensor_to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value.detach().float().cpu().tolist(), dtype=np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class OpenCLIPSigLIP2Embedder:
    def __init__(self, model_name: str, pretrained: str, device: str, torch_dtype: str, image_size: int) -> None:
        self.torch = require_torch()
        try:
            import open_clip
        except Exception as exc:
            raise RuntimeError("Install open_clip_torch to use the SigLIP2 embedder.") from exc
        if device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false. Use --device cpu for a CPU smoke test.")
        self.device = self.torch.device(device)
        self.autocast_dtype = {
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "float32": self.torch.float32,
        }[torch_dtype]
        precision_arg = {
            "float16": "fp16",
            "bfloat16": "bf16",
            "float32": "fp32",
        }[torch_dtype]
        model_source = resolve_open_clip_source(model_name)
        if model_source != model_name:
            print(f"Using cached OpenCLIP snapshot: {model_source.removeprefix('local-dir:')}")
        pretrained_arg = pretrained or None
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_source,
            pretrained=pretrained_arg,
            precision=precision_arg,
            device=self.device,
        )
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_source)
        self.image_size = int(image_size)

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                tokens = self.tokenizer(batch_texts).to(self.device)
                with self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                    feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                chunks.append(torch_tensor_to_numpy(feats))
                print(f"Encoded candidate captions {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return normalize_rows(np.concatenate(chunks, axis=0))

    def encode_images(self, image_paths: list[Path], batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(image_paths), batch_size):
                paths = image_paths[start : start + batch_size]
                tensors = []
                for path in paths:
                    image = Image.open(path).convert("RGB")
                    tensors.append(self.preprocess_image_no_numpy(image))
                batch = self.torch.stack(tensors, dim=0).to(self.device)
                with self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                    feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                chunks.append(torch_tensor_to_numpy(feats))
                print(f"Encoded images {min(start + batch_size, len(image_paths))}/{len(image_paths)}", flush=True)
        return normalize_rows(np.concatenate(chunks, axis=0))

    def encode_image(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            tensor = self.preprocess_image_no_numpy(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            with self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return normalize_rows(torch_tensor_to_numpy(feats))[0]

    def preprocess_image_no_numpy(self, image: Image.Image) -> Any:
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        image = image.convert("RGB").resize((self.image_size, self.image_size), resampling)
        raw = image.tobytes()
        try:
            tensor = self.torch.frombuffer(raw, dtype=self.torch.uint8)
        except Exception:
            storage = self.torch.ByteStorage.from_buffer(raw)
            tensor = self.torch.ByteTensor(storage)
        tensor = tensor.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1).float()
        tensor = tensor / 255.0
        return (tensor - 0.5) / 0.5


class ONNXSigLIP2PathEmbedder:
    def __init__(self, args: argparse.Namespace) -> None:
        from task1_ingredients import ONNXSigLIP2ImageEmbedder

        self.embedder = ONNXSigLIP2ImageEmbedder(args)
        self.provider = self.embedder.provider

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self.embedder.encode_texts(texts, batch_size)

    def encode_image(self, image_path: Path) -> np.ndarray:
        embedding = self.embedder.encode_image(image_path)
        self.last_profile = dict(getattr(self.embedder, "last_profile", {}))
        return embedding

    def encode_images(self, image_paths: list[Path], batch_size: int) -> np.ndarray:
        del batch_size
        return normalize_rows(np.stack([self.encode_image(path) for path in image_paths], axis=0))


def build_siglip2_embedder(args: argparse.Namespace) -> Any:
    if args.siglip_backend == "onnx":
        return ONNXSigLIP2PathEmbedder(args)
    return OpenCLIPSigLIP2Embedder(
        args.siglip_model,
        args.siglip_pretrained,
        args.device,
        args.torch_dtype,
        args.siglip_image_size,
    )


def load_or_build_caption_embeddings(
    embedder: Any,
    caption_bank: list[CaptionBankItem],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_hash = sha256_jsonable(
        {
            "version": CAPTION_CACHE_VERSION,
            "model": args.siglip_model,
            "pretrained": args.siglip_pretrained,
            "caption_text_mode": args.caption_text_mode,
            "captions": [
                {
                    "caption_id": item.caption_id,
                    "cat": item.cat,
                    "filename": item.filename,
                    "caption": item.caption,
                    "text": item.text,
                }
                for item in caption_bank
            ],
        }
    )
    info = {"path": str(args.caption_cache), "enabled": not args.no_caption_cache, "hit": False, "hash": cache_hash}
    if not args.no_caption_cache and args.caption_cache.exists():
        try:
            data = np.load(args.caption_cache, allow_pickle=False)
            if str(data["cache_hash"]) == cache_hash:
                info["hit"] = True
                return normalize_rows(data["caption_embeddings"]), info
        except Exception as exc:
            info["load_warning"] = str(exc)
    texts = [item.text for item in caption_bank]
    embeddings = embedder.encode_texts(texts, args.text_batch_size)
    if not args.no_caption_cache:
        args.caption_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.caption_cache,
            caption_embeddings=embeddings.astype(np.float32),
            cache_version=np.asarray(CAPTION_CACHE_VERSION),
            cache_hash=np.asarray(cache_hash),
        )
    return embeddings, info


def load_image_cache(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    info = {"path": str(args.image_cache), "enabled": not args.no_image_cache, "hit_count": 0}
    if args.no_image_cache or not args.image_cache.exists():
        return {}, info
    try:
        data = np.load(args.image_cache, allow_pickle=False)
        if str(data["cache_version"]) != IMAGE_CACHE_VERSION:
            info["load_warning"] = "cache_version_mismatch"
            return {}, info
        if (
            str(data["siglip_model"]) != args.siglip_model
            or str(data["siglip_pretrained"]) != args.siglip_pretrained
            or int(data["siglip_image_size"]) != int(args.siglip_image_size)
        ):
            info["load_warning"] = "model_mismatch"
            return {}, info
        names = [str(x) for x in data["image_names"].tolist()]
        embeddings = normalize_rows(data["image_embeddings"])
        cache = {name: embeddings[idx] for idx, name in enumerate(names)}
        info["hit_count"] = len(cache)
        return cache, info
    except Exception as exc:
        info["load_warning"] = str(exc)
        return {}, info


def save_image_cache(path: Path, cache: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    if args.no_image_cache:
        return
    names = sorted(cache)
    if not names:
        return
    embeddings = np.stack([cache[name] for name in names], axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.asarray(IMAGE_CACHE_VERSION),
        siglip_model=np.asarray(args.siglip_model),
        siglip_pretrained=np.asarray(args.siglip_pretrained),
        siglip_image_size=np.asarray(int(args.siglip_image_size)),
        image_names=np.asarray(names),
        image_embeddings=embeddings,
    )


def load_or_build_image_embeddings(
    embedder: OpenCLIPSigLIP2Embedder,
    image_names: list[str],
    args: argparse.Namespace,
    image_paths: list[Path] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache, info = load_image_cache(args)
    missing = [name for name in image_names if name not in cache]
    info["requested_count"] = len(image_names)
    info["missing_count"] = len(missing)
    if missing:
        path_by_name = dict(zip(image_names, image_paths or []))
        paths = [path_by_name.get(name, args.image_dir / name) for name in missing]
        encoded = embedder.encode_images(paths, args.image_batch_size)
        for name, row in zip(missing, encoded):
            cache[name] = row.astype(np.float32)
        save_image_cache(args.image_cache, cache, args)
    rows = [cache[name] for name in image_names]
    return normalize_rows(np.stack(rows, axis=0)), info


def build_caption_recall(
    scores: np.ndarray,
    caption_bank: list[CaptionBankItem],
    args: argparse.Namespace,
) -> list[CaptionCandidate]:
    if len(caption_bank) == 0:
        return []
    depth = min(max(args.siglip_top_k, args.rerank_top_k), len(caption_bank))
    top_idx = np.argpartition(-scores, depth - 1)[:depth]
    ordered = top_idx[np.argsort(-scores[top_idx])]
    candidates: list[CaptionCandidate] = []
    for idx in ordered[: args.rerank_top_k]:
        item = caption_bank[int(idx)]
        candidates.append(
            CaptionCandidate(
                caption_id=item.caption_id,
                cat=item.cat,
                filename=item.filename,
                caption=item.caption,
                text=item.text,
                siglip_score=float(scores[int(idx)]),
            )
        )
    return candidates


def patch_qwen_reranker_config(model_name_or_path: str, local_files_only: bool, hf_token: str | None) -> str:
    local_path = Path(model_name_or_path).expanduser()
    if not local_path.exists():
        cached = cached_hf_snapshot(model_name_or_path, RERANKER_REQUIRED_FILES, require_model_weights=True)
        if cached is not None:
            local_path = cached
        else:
            try:
                from huggingface_hub import snapshot_download
            except Exception as exc:
                raise RuntimeError("Install huggingface_hub to download or locate the Qwen reranker model.") from exc
            local_path = Path(snapshot_download(repo_id=model_name_or_path, local_files_only=local_files_only, token=hf_token))
    score_head = local_path / "1_CausalScoreHead"
    score_head.mkdir(parents=True, exist_ok=True)
    cfg_path = score_head / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("true_token_id", 9693)
    cfg.setdefault("false_token_id", 2152)
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    return str(local_path)


def preflight_reranker_requirements(args: argparse.Namespace) -> None:
    if args.mock_reranker or args.final_score_mode == "siglip":
        return
    if args.reranker_backend == "mtmd":
        required = {
            "mtmd bridge": args.reranker_mtmd_executable,
            "reranker GGUF": args.reranker_gguf,
            "multimodal projector": args.reranker_mmproj,
        }
        missing = [f"{label}: {Path(path).expanduser()}" for label, path in required.items() if not Path(path).expanduser().is_file()]
        if missing:
            raise FileNotFoundError("Missing mtmd reranker artifact(s):\n  " + "\n  ".join(missing))
        return
    if args.reranker_backend == "qairt":
        if args.reranker_qairt_bundle is None:
            raise ValueError("--reranker-qairt-bundle is required by --reranker-backend qairt")
        required = {
            "QAIRT reranker worker": args.reranker_qairt_executable,
            "QAIRT reranker bundle": args.reranker_qairt_bundle,
            "QAIRT reranker contract": args.reranker_qairt_contract,
            "QAIRT HTP backend": args.reranker_qairt_backend_path,
        }
        missing = [
            f"{label}: {Path(path).expanduser()}"
            for label, path in required.items()
            if not Path(path).expanduser().exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing native QAIRT reranker artifact(s):\n  " + "\n  ".join(missing)
            )
        return
    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency for Qwen3-VL reranking: sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    model_path = Path(args.reranker_model).expanduser()
    if (
        args.reranker_local_files_only
        and not model_path.exists()
        and cached_hf_snapshot(args.reranker_model, RERANKER_REQUIRED_FILES, require_model_weights=True) is None
    ):
        raise FileNotFoundError(
            f"Qwen reranker is not cached locally: {args.reranker_model}. "
            "Remove --reranker-local-files-only to download it, pass a local model path with --reranker-model, "
            "or choose a cached alias with --reranker-size."
        )


class QwenVLCaptionReranker:
    def __init__(self, args: argparse.Namespace, hf_token: str | None) -> None:
        torch = require_torch()
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError(
                "Qwen3-VL reranker models are loaded through sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the reranker, but torch.cuda.is_available() is false.")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.torch_dtype]
        patched = patch_qwen_reranker_config(args.reranker_model, args.reranker_local_files_only, hf_token)
        model_kwargs: dict[str, Any] = {"torch_dtype": dtype if args.device == "cuda" else torch.float32}
        if args.reranker_attn_implementation:
            model_kwargs["attn_implementation"] = args.reranker_attn_implementation
        print(f"Loading Qwen caption reranker: {patched}")
        try:
            self.model = CrossEncoder(patched, device=args.device, model_kwargs=model_kwargs)
        except Exception as exc:
            if "attn_implementation" in model_kwargs:
                print(f"Retrying reranker load without attn_implementation after {type(exc).__name__}: {exc}")
                model_kwargs.pop("attn_implementation", None)
                self.model = CrossEncoder(patched, device=args.device, model_kwargs=model_kwargs)
            else:
                raise
        self.torch = torch
        self.activation = torch.nn.Sigmoid()
        self.prompt = (
            "Carefully compare the food image with the candidate caption. "
            "Score high only when the visible dish, ingredients, cooking method, sauce, plating, "
            "and other visual details match the caption."
        )

    @staticmethod
    def load_pair_image(path: Path) -> Image.Image:
        if not path.is_file():
            raise FileNotFoundError(f"Reranker image does not exist: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")

    def make_pairs(self, image_paths: list[Path], candidate_texts: list[str], start: int, end: int) -> list[tuple[dict[str, Any], dict[str, str]]]:
        return [
            ({"image": self.load_pair_image(image_paths[idx])}, {"text": str(candidate_texts[idx])})
            for idx in range(start, end)
        ]

    @staticmethod
    def close_pair_images(pairs: list[tuple[dict[str, Any], dict[str, str]]]) -> None:
        for image_part, _ in pairs:
            image = image_part.get("image")
            if hasattr(image, "close"):
                image.close()

    def score_pairs(self, image_paths: list[Path], candidate_texts: list[str], batch_size: int) -> list[float]:
        if len(image_paths) != len(candidate_texts):
            raise ValueError("image_paths and candidate_texts must have the same length")
        scores: list[float] = []
        idx = 0
        batch_size = max(1, int(batch_size))
        total_pairs = len(candidate_texts)
        while idx < total_pairs:
            end = min(idx + batch_size, total_pairs)
            pairs = self.make_pairs(image_paths, candidate_texts, idx, end)
            try:
                with self.torch.inference_mode():
                    out = self.model.predict(
                        pairs,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        activation_fn=self.activation,
                        prompt=self.prompt,
                    )
                scores.extend(float(x) for x in np.asarray(out, dtype=np.float32).reshape(-1).tolist())
                print(f"Reranked image-caption pairs {end}/{total_pairs}", flush=True)
                idx = end
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"Reranker OOM; retrying with batch_size={batch_size}", flush=True)
                    continue
                raise
            finally:
                self.close_pair_images(pairs)
        return scores


class MtmdQwenVLCaptionReranker:
    def __init__(self, args: argparse.Namespace) -> None:
        from model_build.qwen3_vl_reranker import MtmdRerankerClient

        print(
            f"Loading Qwen mtmd caption reranker: {args.reranker_gguf} "
            f"(text compute={args.reranker_mtmd_compute}, "
            f"max candidate batch={args.reranker_mtmd_max_batch_candidates})"
        )
        self.client = MtmdRerankerClient(
            executable=args.reranker_mtmd_executable,
            model=args.reranker_gguf,
            mmproj=args.reranker_mmproj,
            compute=args.reranker_mtmd_compute,
            n_ctx=args.reranker_context_size,
            n_batch=args.reranker_batch_capacity,
            threads=args.reranker_mtmd_threads,
            flash_attention=args.reranker_flash_attention,
            image_max_tokens=args.reranker_image_max_tokens,
            max_batch_candidates=args.reranker_mtmd_max_batch_candidates,
            force_individual_calls=args.reranker_individual_calls,
            image_size=FIXED_IMAGE_SIZE,
            startup_timeout_sec=args.reranker_startup_timeout_sec,
            request_timeout_sec=args.reranker_request_timeout_sec,
            restart_on_failure=not args.no_reranker_restart,
            verbose=args.reranker_mtmd_verbose,
        )
        self.last_profile: dict[str, Any] = {}

    def score_pairs(self, image_paths: list[Path], candidate_texts: list[str], batch_size: int) -> list[float]:
        try:
            profile = self.client.score_pairs_detailed(image_paths, candidate_texts, batch_size)
        finally:
            self.last_profile = dict(self.client.last_profile)
        return [float(score) for score in profile["scores"]]

    def close(self) -> None:
        self.client.close()


def build_caption_reranker(args: argparse.Namespace, hf_token: str | None) -> Any:
    if args.reranker_backend == "mtmd":
        return MtmdQwenVLCaptionReranker(args)
    if args.reranker_backend == "qairt":
        from model_build.qwen3_vl_reranker.qairt_backend import QairtCaptionReranker
        return QairtCaptionReranker(args, image_size=FIXED_IMAGE_SIZE)
    return QwenVLCaptionReranker(args, hf_token)


def reranker_mode_name(args: argparse.Namespace) -> str:
    if args.reranker_backend == "mtmd":
        base = f"mtmd:{args.reranker_mtmd_compute}:{args.reranker_gguf}"
    elif args.reranker_backend == "qairt":
        base = f"qairt:{args.reranker_qairt_bundle}"
    else:
        base = f"transformers:{args.reranker_model}"
    return base if args.final_score_mode != "siglip_guarded" else f"siglip_guarded:{base}"


def normalize_score_values(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    std = float(arr.std())
    if std < 1e-12:
        return [0.0 for _ in values]
    return ((arr - float(arr.mean())) / std).astype(np.float32).tolist()


def siglip_top_gap(candidates: list[CaptionCandidate]) -> float:
    """Return the SigLIP score margin between the best and second-best captions."""
    if len(candidates) <= 1:
        return float("inf")
    ranked = sorted(candidates, key=lambda c: (-float(c.siglip_score), c.filename, c.caption_id))
    return float(ranked[0].siglip_score - ranked[1].siglip_score)


def siglip_is_confident(candidates: list[CaptionCandidate], args: argparse.Namespace) -> bool:
    return siglip_top_gap(candidates) >= float(args.siglip_keep_gap)


def final_candidate_scores(candidates: list[CaptionCandidate], args: argparse.Namespace) -> dict[int, float]:
    if not candidates:
        return {}
    if args.final_score_mode == "siglip":
        return {idx: float(candidate.siglip_score) for idx, candidate in enumerate(candidates)}
    if args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args):
        return {idx: float(candidate.siglip_score) for idx, candidate in enumerate(candidates)}

    rerank_values = [float(candidate.rerank_score if candidate.rerank_score is not None else candidate.siglip_score) for candidate in candidates]
    if args.final_score_mode in {"rerank", "siglip_guarded"}:
        return {idx: rerank_values[idx] for idx in range(len(candidates))}
    if args.final_score_mode == "blend_z":
        siglip_z = normalize_score_values([float(candidate.siglip_score) for candidate in candidates])
        rerank_z = normalize_score_values(rerank_values)
        weight = min(1.0, max(0.0, float(args.rerank_weight)))
        return {idx: float(weight * rerank_z[idx] + (1.0 - weight) * siglip_z[idx]) for idx in range(len(candidates))}
    raise ValueError(f"Unsupported final-score mode: {args.final_score_mode}")


def candidate_to_dict(candidate: CaptionCandidate, final_score: float | None = None) -> dict[str, Any]:
    return {
        "caption_id": candidate.caption_id,
        "cat": candidate.cat,
        "filename": candidate.filename,
        "caption": candidate.caption,
        "text": candidate.text,
        "siglip_score": candidate.siglip_score,
        "rerank_score": candidate.rerank_score,
        "final_score": final_score,
    }


def row_accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def build_artifact_manifest(args: argparse.Namespace) -> dict[str, Any]:
    siglip_contexts: list[dict[str, Any]] = []
    if args.siglip_backend == "onnx":
        first = artifact_record(args.siglip_onnx_path)
        if first is not None:
            siglip_contexts.append(first)
        if args.siglip_context_mode == "split":
            second = artifact_record(args.siglip_onnx_stage2_path)
            if second is not None:
                siglip_contexts.append(second)
    return {
        "siglip_contexts": siglip_contexts,
        "caption_embeddings": artifact_record(args.caption_cache),
        "reranker_model": (
            artifact_record(args.reranker_gguf)
            if args.reranker_backend == "mtmd" and not args.mock_reranker
            else None
        ),
        "reranker_projector": (
            artifact_record(args.reranker_mmproj)
            if args.reranker_backend == "mtmd" and not args.mock_reranker
            else None
        ),
    }


def summarize_detailed_profiles(traces: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    siglip_profiles = [
        row.get("profiling", {}).get("siglip", {})
        for row in traces
        if row.get("profiling", {}).get("siglip")
    ]
    reranker_profiles = [
        row.get("profiling", {}).get("reranker", {})
        for row in traces
        if row.get("rerank_applied")
    ]
    pair_results = [
        result
        for profile in reranker_profiles
        for result in profile.get("results", [])
    ]
    max_positions = max(
        (int(result.get("positions", result.get("tokens", 0))) for result in pair_results),
        default=0,
    )
    placements = {
        json.dumps(result.get("backend_placement", {}), sort_keys=True)
        for result in pair_results
    }
    fallbacks = sorted(
        {
            str(operation)
            for result in pair_results
            for operation in result.get("fallback_operations", [])
        }
    )
    return {
        "siglip": {
            "rows": len(siglip_profiles),
            "components_sec": component_summary(siglip_profiles, SIGLIP_COMPONENTS),
        },
        "reranker": {
            "conditional_image_count": len(reranker_profiles),
            "pair_count": len(pair_results),
            "components_ms": component_summary(pair_results, RERANKER_COMPONENTS_MS),
            "maximum_sequence_tokens": max_positions,
            "sequence_token_count_scope": "bridge_reported_candidate_positions_only",
            "arithmetic_context_size": recommended_context_capacity(max_positions),
            "recommended_context_size": max(
                1024, recommended_context_capacity(max_positions)
            ),
            "recommendation_basis": "empirical_safe_floor_pending_total_context_telemetry",
            "configured_context_size": int(args.reranker_context_size),
            "configured_batch_capacity": int(args.reranker_batch_capacity),
            "flash_attention": args.reranker_flash_attention,
            "individual_calls": bool(args.reranker_individual_calls or args.reranker_backend == "qairt"),
            "backend_placements": [json.loads(value) for value in sorted(placements)],
            "fallback_operations": fallbacks,
            "cache_hits": {
                "vision": sum(bool(result.get("vision_cache_hit")) for result in pair_results),
                "system_prefix": sum(bool(result.get("system_prefix_cache_hit")) for result in pair_results),
                "image_kv": sum(bool(result.get("image_kv_cache_hit")) for result in pair_results),
            },
        },
        "peak_memory": process_memory_snapshot(),
    }


def run_pipeline_per_query_measurement(args: argparse.Namespace) -> dict[str, Any]:
    t0_total = time.perf_counter()
    run_config = base_run_config(ROOT)
    evk_stack = (
        load_and_check_evk_stack(
            args.evk_stack_manifest,
            selected_qnn_backend_path=args.siglip_qnn_backend_path,
        )
        if args.evk_stack_manifest is not None and args.evk_stack_manifest.is_file()
        else None
    )
    if (
        args.strict_evk_stack
        and evk_stack is not None
        and not evk_stack["assessment"]["compatible"]
    ):
        raise SystemExit(
            "EVK QAIRT stack contract failed; run scripts/check_evk_stack.py for details."
        )
    hf_token = configure_hf_token(args)

    caption_bank_json = args.caption_bank_json or args.evaluation_json

    if args.inference_only:
        evaluation_rows: list[Food500Row] = []
        manifest_rows: list[Food500Row] = []
        gt_by_filename: dict[str, Food500Row] = {}
        caption_bank = read_caption_bank_items(caption_bank_json, args.caption_text_mode)
    else:
        evaluation_rows = read_food500_rows(args.evaluation_json)
        manifest_rows = read_manifest(args.manifest)
        gt_by_filename = {canonical_filename(row.filename): row for row in evaluation_rows}
        caption_bank = read_caption_bank_items(caption_bank_json, args.caption_text_mode)
    print(f"Candidate captions: {len(caption_bank)}")
    preflight_reranker_requirements(args)

    single_image_name = None
    if args.image is not None:
        single_image_name, single_image_path = resolve_image_name(args.image, args.image_dir)
        target_rows = [gt_by_filename.get(canonical_filename(single_image_name), Food500Row(cat="", filename=single_image_name, caption=""))]
        image_paths = [single_image_path]
    else:
        target_rows = select_inference_rows(args) if args.inference_only else select_eval_rows(args, evaluation_rows, manifest_rows)
        resolved_names, missing = resolve_image_list_names(
            args.image_dir, [row.filename for row in target_rows]
        )
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} selected images are missing from {args.image_dir}: {missing[:10]}"
            )
        image_paths = [args.image_dir / name for name in resolved_names]
    image_names = [row.filename for row in target_rows]
    contract_payload = {
        "fixed_image_resolution": [FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE],
        "images": image_names,
        "resolved_image_paths": [str(path.resolve()) for path in image_paths],
        "caption_bank_json": str(caption_bank_json.resolve()),
        "caption_text_mode": args.caption_text_mode,
        "siglip_context_mode": args.siglip_context_mode,
        "siglip_stage1": str(args.siglip_onnx_path),
        "siglip_stage2": str(args.siglip_onnx_stage2_path),
        "reranker_gguf": str(args.reranker_gguf),
        "reranker_mmproj": str(args.reranker_mmproj),
        "reranker_context_size": args.reranker_context_size,
        "reranker_batch_capacity": args.reranker_batch_capacity,
        "rerank_top_k": args.rerank_top_k,
        "final_score_mode": args.final_score_mode,
        "siglip_keep_gap": args.siglip_keep_gap,
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(contract_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_contract = {"sha256": contract_sha256, "payload": contract_payload}
    prior_checkpoint: dict[str, Any] = {}
    if args.resume:
        if args.checkpoint_json is None or not args.checkpoint_json.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_json}")
        prior_checkpoint = json.loads(args.checkpoint_json.read_text(encoding="utf-8"))
        if prior_checkpoint.get("run_contract", {}).get("sha256") != contract_sha256:
            raise ValueError("Checkpoint run contract does not match the requested model/data config")

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    embedder = build_siglip2_embedder(args)
    timings["load_siglip2_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    caption_embeddings, caption_cache = load_or_build_caption_embeddings(embedder, caption_bank, args)
    timings["caption_embeddings_sec"] = time.perf_counter() - t0

    reranker: QwenVLCaptionReranker | MtmdQwenVLCaptionReranker | None = None
    t0 = time.perf_counter()
    if args.mock_reranker or args.final_score_mode == "siglip":
        reranker_mode = "mock_siglip_scores" if args.mock_reranker else "skipped_siglip_final_score"
    else:
        reranker = build_caption_reranker(args, hf_token)
        reranker_mode = reranker_mode_name(args)
    timings["load_reranker_sec"] = time.perf_counter() - t0

    power_monitor = PowerMonitor(enabled=args.measure_power, interval_ms=args.power_sample_interval_ms)
    power_monitor.start()
    measurement_t0 = time.perf_counter()
    measured_wall_sec = 0.0
    power_summary: dict[str, Any] = {}

    traces: list[dict[str, Any]] = [
        dict(trace)
        for trace in prior_checkpoint.get("traces", [])
        if trace.get("status", "ok") == "ok"
    ]
    csv_rows: list[dict[str, Any]] = [
        dict(row) for row in prior_checkpoint.get("csv_rows", [])
    ]
    failure_events: list[dict[str, Any]] = [
        dict(event) for event in prior_checkpoint.get("failure_events", [])
    ]
    completed_images = {str(trace.get("image")) for trace in traces}
    correct_caption = sum(int(bool(trace.get("caption_correct"))) for trace in traces)
    correct_class = sum(int(bool(trace.get("class_correct"))) for trace in traces)
    siglip_correct_caption = sum(
        int(bool(trace.get("siglip_top1_caption_correct"))) for trace in traces
    )
    truth_caption_in_candidates = sum(
        int(bool(trace.get("truth_caption_in_candidates"))) for trace in traces
    )
    rerank_pair_count = sum(int(trace.get("rerank_candidate_count", 0)) for trace in traces)
    rerank_skipped_by_siglip_guard = sum(
        int(not trace.get("rerank_applied")) for trace in traces
    )
    latencies_sec: list[float] = []
    session_success_count = 0
    worker_restart_count = 0

    def write_checkpoint() -> None:
        if args.checkpoint_json is None:
            return
        atomic_write_json(
            args.checkpoint_json,
            {
                "schema_version": "dishcovery_task2_checkpoint_v1",
                "run_contract": run_contract,
                "resumed": bool(args.resume),
                "complete": len(completed_images) == len(target_rows),
                "requested_rows": len(target_rows),
                "successful_rows": len(completed_images),
                "failure_event_count": len(failure_events),
                "worker_restart_count": worker_restart_count,
                "traces": traces,
                "csv_rows": csv_rows,
                "failure_events": failure_events,
            },
        )

    write_checkpoint()
    try:
        for idx, (row, image_path) in enumerate(zip(target_rows, image_paths), start=1):
            if row.filename in completed_images:
                print(f"[{idx}/{len(target_rows)}] {row.filename} resumed checkpoint hit", flush=True)
                continue
            query_t0 = time.perf_counter()

            t0 = time.perf_counter()
            image_embedding = embedder.encode_image(image_path)
            siglip_profile = dict(getattr(embedder, "last_profile", {}))
            siglip_profile["timings_sec"] = dict(siglip_profile.get("timings_sec", {}))
            recall_t0 = time.perf_counter()
            scores = 100.0 * (image_embedding @ caption_embeddings.T)
            siglip_profile["timings_sec"]["cosine_recall_sec"] = time.perf_counter() - recall_t0
            topk_t0 = time.perf_counter()
            candidates = build_caption_recall(scores, caption_bank, args)
            siglip_profile["timings_sec"]["topk_selection_sec"] = time.perf_counter() - topk_t0
            siglip_image_and_recall_sec = time.perf_counter() - t0
            siglip_profile["timings_sec"]["total_siglip_sec"] = siglip_image_and_recall_sec

            t0 = time.perf_counter()
            rerank_applied = False
            reranker_profile: dict[str, Any] = {
                "skipped": True,
                "reason": "not_requested_or_guarded",
                "results": [],
                "requests": [],
            }
            if args.mock_reranker or args.final_score_mode == "siglip":
                for candidate in candidates:
                    candidate.rerank_score = candidate.siglip_score
            elif args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args):
                rerank_skipped_by_siglip_guard += 1
            else:
                if reranker is None:
                    raise RuntimeError("Reranker was not initialized")
                pair_paths = [image_path for _ in candidates]
                pair_texts = [candidate.text for candidate in candidates]
                rerank_pair_count += len(pair_paths)
                rerank_applied = True
                try:
                    rerank_scores = reranker.score_pairs(
                        pair_paths, pair_texts, args.reranker_batch_size
                    )
                except (MtmdWorkerError, QairtWorkerError) as exc:
                    rerank_pair_count -= len(pair_paths)
                    partial_profile = dict(getattr(reranker, "last_profile", {}))
                    failure_events.append(
                        {
                            "image": row.filename,
                            "image_path": str(image_path),
                            "stage": "reranker",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "worker_restarted": bool(exc.restarted),
                            "completed_candidate_count": int(
                                partial_profile.get("completed_candidate_count", 0)
                            ),
                            "elapsed_sec": time.perf_counter() - query_t0,
                            "profiling": {
                                "siglip": siglip_profile,
                                "reranker": partial_profile,
                                "peak_memory": process_memory_snapshot(),
                            },
                        }
                    )
                    write_checkpoint()
                    print(
                        f"[{idx}/{len(target_rows)}] {row.filename} FAILED: {exc}",
                        flush=True,
                    )
                    continue
                reranker_profile = dict(getattr(reranker, "last_profile", {}))
                reranker_profile["skipped"] = False
                for candidate, score in zip(candidates, rerank_scores):
                    candidate.rerank_score = score
            rerank_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            final_scores = final_candidate_scores(candidates, args)
            ranked = sorted(
                enumerate(candidates),
                key=lambda item: (
                    -float(final_scores[int(item[0])]),
                    -float(item[1].siglip_score),
                    item[1].filename,
                    item[1].caption_id,
                ),
            )
            ranked_candidates = [candidate for _, candidate in ranked]
            ranked_scores = {id(candidate): float(final_scores[candidate_idx]) for candidate_idx, candidate in ranked}
            siglip_ranked = sorted(candidates, key=lambda c: (-float(c.siglip_score), c.filename, c.caption_id))

            pred = ranked_candidates[0] if ranked_candidates else None
            siglip_pred = siglip_ranked[0] if siglip_ranked else None

            is_caption_correct = None if args.inference_only else bool(pred and row.filename and pred.filename == row.filename)
            is_class_correct = None if args.inference_only else bool(pred and row.cat and pred.cat == row.cat)
            is_siglip_caption_correct = None if args.inference_only else bool(siglip_pred and row.filename and siglip_pred.filename == row.filename)
            has_truth_caption = None if args.inference_only else (any(candidate.filename == row.filename for candidate in candidates) if row.filename else False)

            if not args.inference_only:
                correct_caption += int(bool(is_caption_correct))
                correct_class += int(bool(is_class_correct))
                siglip_correct_caption += int(bool(is_siglip_caption_correct))
                truth_caption_in_candidates += int(bool(has_truth_caption))
            final_selection_sec = time.perf_counter() - t0
            total_image_sec = time.perf_counter() - query_t0
            latencies_sec.append(total_image_sec)

            trace = {
                "status": "ok",
                "image": row.filename,
                "image_path": str(image_path),
                "truth_cat": row.cat,
                "truth_caption": row.caption,
                "prediction_filename": pred.filename if pred else "",
                "prediction_cat": pred.cat if pred else "",
                    "prediction_caption": pred.caption if pred else "",
                    "caption_correct": is_caption_correct,
                    "class_correct": is_class_correct,
                "siglip_top1_filename": siglip_pred.filename if siglip_pred else "",
                "siglip_top1_caption": siglip_pred.caption if siglip_pred else "",
                "siglip_top1_caption_correct": is_siglip_caption_correct,
                "truth_caption_in_candidates": has_truth_caption,
                "siglip_top_gap": siglip_top_gap(candidates),
                "rerank_applied": rerank_applied,
                "rerank_candidate_count": len(candidates) if rerank_applied else 0,
                "profiling": {
                    "siglip": siglip_profile,
                    "reranker": reranker_profile,
                    "peak_memory": process_memory_snapshot(),
                },
                "timings_sec": {
                    "siglip_image_and_recall_sec": siglip_image_and_recall_sec,
                    "rerank_sec": rerank_sec,
                    "final_selection_sec": final_selection_sec,
                    "total_image_sec": total_image_sec,
                },
                "ranked_candidates": [candidate_to_dict(candidate, ranked_scores[id(candidate)]) for candidate in ranked_candidates],
            }
            traces.append(trace)
            completed_images.add(row.filename)
            session_success_count += 1
            csv_rows.append(
                {
                    "image": row.filename,
                    "truth_cat": row.cat,
                    "truth_caption": row.caption,
                    "prediction_filename": pred.filename if pred else "",
                    "prediction_cat": pred.cat if pred else "",
                    "prediction_caption": pred.caption if pred else "",
                    "caption_correct": "" if is_caption_correct is None else int(is_caption_correct),
                    "class_correct": "" if is_class_correct is None else int(is_class_correct),
                    "siglip_top1_filename": siglip_pred.filename if siglip_pred else "",
                    "siglip_top1_caption_correct": "" if is_siglip_caption_correct is None else int(is_siglip_caption_correct),
                    "truth_caption_in_candidates": "" if has_truth_caption is None else int(has_truth_caption),
                    "siglip_top_gap": f"{siglip_top_gap(candidates):.6f}",
                    "rerank_applied": int(rerank_applied),
                    "prediction_rerank_score": f"{float(pred.rerank_score if pred and pred.rerank_score is not None else 0.0):.6f}",
                    "prediction_siglip_score": f"{float(pred.siglip_score if pred else 0.0):.6f}",
                    "prediction_final_score": f"{float(ranked_scores.get(id(pred), 0.0) if pred else 0.0):.6f}",
                    "siglip_image_and_recall_sec": f"{siglip_image_and_recall_sec:.6f}",
                    "rerank_sec": f"{rerank_sec:.6f}",
                    "final_selection_sec": f"{final_selection_sec:.6f}",
                    "total_image_sec": f"{total_image_sec:.6f}",
                    "ranked_candidates": "; ".join(f"{candidate.filename}:{ranked_scores[id(candidate)]:.4f}" for candidate in ranked_candidates),
                }
            )
            write_checkpoint()
            print(
                f"[{idx}/{len(target_rows)}] {row.filename} "
                f"{'prediction=' + pred.caption[:80] if args.inference_only and pred else 'caption_top1=' + str(int(bool(is_caption_correct))) + ' class_top1=' + str(int(bool(is_class_correct)))} "
                f"time={total_image_sec:.3f}s rerank={rerank_sec:.3f}s"
            )
    finally:
        measured_wall_sec = time.perf_counter() - measurement_t0
        power_summary = power_monitor.stop(wall_sec=measured_wall_sec)
        if reranker is not None and hasattr(reranker, "client"):
            worker_restart_count = int(reranker.client.restart_count)
        write_checkpoint()
        if reranker is not None and hasattr(reranker, "close"):
            reranker.close()

    requested_total = len(target_rows)
    total = len(traces)
    if args.inference_only:
        metrics = {
            "rows": total,
            "requested_rows": requested_total,
            "failed_rows": requested_total - total,
            "failure_event_count": len(failure_events),
            "metric_available": False,
            "caption_accuracy": None,
            "caption_correct": None,
            "class_accuracy": None,
            "class_correct": None,
            "siglip_top1_caption_accuracy": None,
            "siglip_top1_caption_correct": None,
            "truth_caption_in_rerank_candidates_rate": None,
            "truth_caption_in_rerank_candidates": None,
            "rerank_pair_count": rerank_pair_count,
            "rerank_skipped_by_siglip_guard": rerank_skipped_by_siglip_guard,
        }
    else:
        metrics = {
            "rows": total,
            "requested_rows": requested_total,
            "failed_rows": requested_total - total,
            "failure_event_count": len(failure_events),
            "metric_available": True,
            "caption_accuracy": row_accuracy(correct_caption, total),
            "caption_correct": correct_caption,
            "class_accuracy": row_accuracy(correct_class, total),
            "class_correct": correct_class,
            "siglip_top1_caption_accuracy": row_accuracy(siglip_correct_caption, total),
            "siglip_top1_caption_correct": siglip_correct_caption,
            "truth_caption_in_rerank_candidates_rate": row_accuracy(truth_caption_in_candidates, total),
            "truth_caption_in_rerank_candidates": truth_caption_in_candidates,
            "rerank_pair_count": rerank_pair_count,
            "rerank_skipped_by_siglip_guard": rerank_skipped_by_siglip_guard,
        }
    timings["measured_query_wall_sec"] = measured_wall_sec
    timings["total_sec"] = time.perf_counter() - t0_total

    benchmark = build_benchmark_summary(
        task_name="task2_caption_inference_only" if args.inference_only else "task2_food500_caption",
        query_count=session_success_count,
        latencies_sec=latencies_sec,
        measured_wall_sec=measured_wall_sec,
        task_metric_name="not_available_no_ground_truth" if args.inference_only else "top1_caption_accuracy",
        task_metric_value=None if args.inference_only else float(metrics["caption_accuracy"]),
        power_summary=power_summary,
        w_config=args.w_config,
        nvpmodel=query_nvpmodel(),
        extra_task_metrics={
            "metric_available": not args.inference_only,
            "class_top1_accuracy": metrics["class_accuracy"],
            "siglip_top1_caption_accuracy": metrics["siglip_top1_caption_accuracy"],
            "truth_caption_in_rerank_candidates_rate": metrics["truth_caption_in_rerank_candidates_rate"],
        },
    )
    profiling = summarize_detailed_profiles(traces, args)

    return {
        "schema_version": "orin_task2_caption_reranker_per_query_measurement_v3",
        "run_config": run_config,
        "run_contract": run_contract,
        "execution": {
            "complete": total == requested_total,
            "clean_run": total == requested_total and not failure_events and not args.resume,
            "resumed": bool(args.resume),
            "checkpoint_json": str(args.checkpoint_json) if args.checkpoint_json else None,
            "requested_rows": requested_total,
            "successful_rows": total,
            "session_successful_rows": session_success_count,
            "failure_event_count": len(failure_events),
            "failure_events": failure_events,
            "worker_restart_count": worker_restart_count,
        },
        "evk_stack": evk_stack,
        "models": {
            "siglip_model": args.siglip_model,
            "siglip_pretrained": args.siglip_pretrained,
            "siglip_backend": args.siglip_backend,
            "siglip_context_mode": args.siglip_context_mode,
            "siglip_onnx_path": str(args.siglip_onnx_path) if args.siglip_backend == "onnx" else "",
            "siglip_onnx_stage2_path": (
                str(args.siglip_onnx_stage2_path)
                if args.siglip_backend == "onnx" and args.siglip_context_mode == "split"
                else ""
            ),
            "siglip_onnx_provider": args.siglip_onnx_provider if args.siglip_backend == "onnx" else "",
            "siglip_precision": "fp16_compiled" if args.siglip_backend == "onnx" else args.torch_dtype,
            "reranker_size": args.reranker_size,
            "reranker_model": args.reranker_model,
            "reranker_backend": args.reranker_backend,
            "reranker_gguf": str(args.reranker_gguf) if args.reranker_backend == "mtmd" else "",
            "reranker_mmproj": str(args.reranker_mmproj) if args.reranker_backend == "mtmd" else "",
            "reranker_mtmd_compute": args.reranker_mtmd_compute if args.reranker_backend == "mtmd" else "",
            "reranker_mtmd_max_batch_candidates": args.reranker_mtmd_max_batch_candidates if args.reranker_backend == "mtmd" else None,
            "reranker_qairt_bundle": str(args.reranker_qairt_bundle) if args.reranker_backend == "qairt" and args.reranker_qairt_bundle else "",
            "reranker_qairt_contract": str(args.reranker_qairt_contract) if args.reranker_backend == "qairt" else "",
            "reranker_qairt_worker": str(args.reranker_qairt_executable) if args.reranker_backend == "qairt" else "",
            "reranker_qairt_backend_path": str(args.reranker_qairt_backend_path) if args.reranker_backend == "qairt" else "",
            "reranker_context_size": args.reranker_context_size,
            "reranker_batch_capacity": args.reranker_batch_capacity,
            "reranker_flash_attention": args.reranker_flash_attention,
            "reranker_mode": reranker_mode,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
        },
        "settings": {
            "caption_text_mode": args.caption_text_mode,
            "siglip_image_size": args.siglip_image_size,
            "siglip_context_mode": args.siglip_context_mode,
            "siglip_io_binding": args.siglip_io_binding,
            "siglip_qnn_backend_path": (
                str(args.siglip_qnn_backend_path.expanduser().resolve())
                if args.siglip_qnn_backend_path is not None
                else None
            ),
            "siglip_qnn_shared_memory_allocator": args.siglip_qnn_shared_memory_allocator,
            "siglip_qnn_finalization_mode": args.siglip_qnn_finalization_mode,
            "siglip_top_k": args.siglip_top_k,
            "rerank_top_k": args.rerank_top_k,
            "final_score_mode": args.final_score_mode,
            "rerank_weight": args.rerank_weight,
            "siglip_keep_gap": args.siglip_keep_gap,
            "reranker_mtmd_max_batch_candidates": args.reranker_mtmd_max_batch_candidates,
            "reranker_individual_calls": args.reranker_individual_calls,
            "reranker_context_size": args.reranker_context_size,
            "reranker_batch_capacity": args.reranker_batch_capacity,
            "reranker_flash_attention": args.reranker_flash_attention,
            "reranker_image_size": FIXED_IMAGE_SIZE,
            "reranker_startup_timeout_sec": args.reranker_startup_timeout_sec,
            "reranker_request_timeout_sec": args.reranker_request_timeout_sec,
            "reranker_restart_on_failure": not args.no_reranker_restart,
            "strict_evk_stack": args.strict_evk_stack,
            "checkpoint_json": str(args.checkpoint_json) if args.checkpoint_json else None,
            "resumed": bool(args.resume),
            "eval_first": args.eval_first,
            "eval_all": args.eval_all,
            "seed": args.seed,
            "measure": True,
            "inference_only": args.inference_only,
        },
        "paths": {
            "image_dir": str(args.image_dir),
            "manifest": str(args.manifest),
            "images_list": str(args.images_list) if args.images_list is not None else "",
            "evaluation_json": str(args.evaluation_json),
            "caption_bank_json": str(caption_bank_json),
            "evk_stack_manifest": str(args.evk_stack_manifest),
        },
        "caption_bank_count": len(caption_bank),
        "caption_cache": caption_cache,
        "image_cache": {"enabled": False, "reason": "per_query_measurement_encodes_each_image"},
        "protocol": {
            "input_images_sequential": True,
            "input_image_embedding_cache": False,
            "candidate_batching": (
                False
                if args.reranker_backend == "qairt"
                else not args.reranker_individual_calls
                if args.reranker_backend == "mtmd"
                else args.reranker_batch_size > 1
            ),
            "candidate_calls_per_reranked_image": (
                args.rerank_top_k
                if args.reranker_individual_calls or args.reranker_backend == "qairt"
                else None
            ),
            "image_kv_prefix_reused_within_image": args.reranker_backend in {"mtmd", "qairt"},
            "model_resident_during_inference": True,
            "phase_separated": False,
            "siglip_input_resolution": [FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE],
            "reranker_input_resolution": [FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE],
        },
        "artifacts": build_artifact_manifest(args),
        "metrics": metrics,
        "benchmark": benchmark,
        "profiling": profiling,
        "timings_sec": timings,
        "traces": traces,
        "csv_rows": csv_rows,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.measure or args.measure_power:
        return run_pipeline_per_query_measurement(args)
    if args.inference_only:
        raise SystemExit("--inference-only currently requires --measure or --measure-power.")

    t0_total = time.perf_counter()
    hf_token = configure_hf_token(args)

    evaluation_rows = read_food500_rows(args.evaluation_json)
    caption_bank_json = args.caption_bank_json or args.evaluation_json
    caption_bank_rows = read_food500_rows(caption_bank_json)
    manifest_rows = read_manifest(args.manifest)

    gt_by_filename = {canonical_filename(row.filename): row for row in evaluation_rows}
    caption_bank = build_caption_bank(caption_bank_rows, args.caption_text_mode)
    print(f"Candidate captions: {len(caption_bank)}")
    preflight_reranker_requirements(args)

    single_image_name = None
    if args.image is not None:
        single_image_name, single_image_path = resolve_image_name(args.image, args.image_dir)
        target_rows = [gt_by_filename.get(canonical_filename(single_image_name), Food500Row(cat="", filename=single_image_name, caption=""))]
        image_paths = [single_image_path]
    else:
        target_rows = select_eval_rows(args, evaluation_rows, manifest_rows)
        resolved_names, missing = resolve_image_list_names(
            args.image_dir, [row.filename for row in target_rows]
        )
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} selected images are missing from {args.image_dir}: {missing[:10]}"
            )
        image_paths = [args.image_dir / name for name in resolved_names]
    image_names = [row.filename for row in target_rows]
    contract_payload = {
        "fixed_image_resolution": [FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE],
        "images": image_names,
        "resolved_image_paths": [str(path.resolve()) for path in image_paths],
        "caption_bank_json": str(caption_bank_json.resolve()),
        "caption_text_mode": args.caption_text_mode,
        "siglip_context_mode": args.siglip_context_mode,
        "siglip_stage1": str(args.siglip_onnx_path),
        "siglip_stage2": str(args.siglip_onnx_stage2_path),
        "reranker_gguf": str(args.reranker_gguf),
        "reranker_mmproj": str(args.reranker_mmproj),
        "reranker_context_size": args.reranker_context_size,
        "reranker_batch_capacity": args.reranker_batch_capacity,
        "rerank_top_k": args.rerank_top_k,
        "final_score_mode": args.final_score_mode,
        "siglip_keep_gap": args.siglip_keep_gap,
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(contract_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_contract = {"sha256": contract_sha256, "payload": contract_payload}
    prior_checkpoint: dict[str, Any] = {}
    if args.resume:
        if args.checkpoint_json is None or not args.checkpoint_json.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_json}")
        prior_checkpoint = json.loads(args.checkpoint_json.read_text(encoding="utf-8"))
        if prior_checkpoint.get("run_contract", {}).get("sha256") != contract_sha256:
            raise ValueError("Checkpoint run contract does not match the requested model/data config")

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    embedder = build_siglip2_embedder(args)
    timings["load_siglip2_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    caption_embeddings, caption_cache = load_or_build_caption_embeddings(embedder, caption_bank, args)
    timings["caption_embeddings_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    image_embeddings, image_cache = load_or_build_image_embeddings(
        embedder, image_names, args, image_paths
    )
    timings["image_embeddings_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    score_matrix = 100.0 * (image_embeddings @ caption_embeddings.T)
    recall_by_image: dict[str, list[CaptionCandidate]] = {}
    for row_idx, image_name in enumerate(image_names):
        recall_by_image[image_name] = build_caption_recall(score_matrix[row_idx], caption_bank, args)
    timings["siglip_caption_recall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rerank_pair_count = 0
    rerank_skipped_by_siglip_guard = 0
    if args.mock_reranker or args.final_score_mode == "siglip":
        for candidates in recall_by_image.values():
            for candidate in candidates:
                candidate.rerank_score = candidate.siglip_score
        reranker_mode = "mock_siglip_scores" if args.mock_reranker else "skipped_siglip_final_score"
        if args.final_score_mode == "siglip":
            rerank_skipped_by_siglip_guard = len(recall_by_image)
    else:
        pair_paths: list[Path] = []
        pair_texts: list[str] = []
        pair_meta: list[tuple[str, int]] = []
        for image_name, image_path in zip(image_names, image_paths):
            candidates = recall_by_image[image_name]
            if args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args):
                rerank_skipped_by_siglip_guard += 1
                continue
            for cand_idx, candidate in enumerate(candidates):
                pair_paths.append(image_path)
                pair_texts.append(candidate.text)
                pair_meta.append((image_name, cand_idx))
        rerank_pair_count = len(pair_paths)
        if pair_paths:
            reranker = build_caption_reranker(args, hf_token)
            scores = reranker.score_pairs(pair_paths, pair_texts, args.reranker_batch_size)
            for (image_name, cand_idx), score in zip(pair_meta, scores):
                recall_by_image[image_name][cand_idx].rerank_score = score
            reranker_mode = reranker_mode_name(args)
        else:
            reranker_mode = "skipped_all_by_siglip_guard" if args.final_score_mode == "siglip_guarded" else "skipped_no_pairs"
    timings["rerank_sec"] = time.perf_counter() - t0

    traces: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    correct_caption = 0
    correct_class = 0
    siglip_correct_caption = 0
    truth_caption_in_candidates = 0

    for row in target_rows:
        candidates = recall_by_image[row.filename]
        final_scores = final_candidate_scores(candidates, args)
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (
                -float(final_scores[int(item[0])]),
                -float(item[1].siglip_score),
                item[1].filename,
                item[1].caption_id,
            ),
        )
        ranked_candidates = [candidate for _, candidate in ranked]
        ranked_scores = {id(candidate): float(final_scores[idx]) for idx, candidate in ranked}
        siglip_ranked = sorted(candidates, key=lambda c: (-float(c.siglip_score), c.filename, c.caption_id))

        pred = ranked_candidates[0] if ranked_candidates else None
        siglip_pred = siglip_ranked[0] if siglip_ranked else None

        is_caption_correct = bool(pred and row.filename and pred.filename == row.filename)
        is_class_correct = bool(pred and row.cat and pred.cat == row.cat)
        is_siglip_caption_correct = bool(siglip_pred and row.filename and siglip_pred.filename == row.filename)
        has_truth_caption = any(candidate.filename == row.filename for candidate in candidates) if row.filename else False

        correct_caption += int(is_caption_correct)
        correct_class += int(is_class_correct)
        siglip_correct_caption += int(is_siglip_caption_correct)
        truth_caption_in_candidates += int(has_truth_caption)

        traces.append(
            {
                "image": row.filename,
                "image_path": str(args.image_dir / row.filename) if single_image_name is None else str(image_paths[0]),
                "truth_cat": row.cat,
                "truth_caption": row.caption,
                "prediction_filename": pred.filename if pred else "",
                "prediction_cat": pred.cat if pred else "",
                "prediction_caption": pred.caption if pred else "",
                "caption_correct": is_caption_correct,
                "class_correct": is_class_correct,
                "siglip_top1_filename": siglip_pred.filename if siglip_pred else "",
                "siglip_top1_caption": siglip_pred.caption if siglip_pred else "",
                "siglip_top1_caption_correct": is_siglip_caption_correct,
                "truth_caption_in_candidates": has_truth_caption,
                "siglip_top_gap": siglip_top_gap(candidates),
                "rerank_applied": not (args.mock_reranker or args.final_score_mode == "siglip" or (args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args))),
                "ranked_candidates": [candidate_to_dict(candidate, ranked_scores[id(candidate)]) for candidate in ranked_candidates],
            }
        )
        csv_rows.append(
            {
                "image": row.filename,
                "truth_cat": row.cat,
                "truth_caption": row.caption,
                "prediction_filename": pred.filename if pred else "",
                "prediction_cat": pred.cat if pred else "",
                "prediction_caption": pred.caption if pred else "",
                "caption_correct": int(is_caption_correct),
                "class_correct": int(is_class_correct),
                "siglip_top1_filename": siglip_pred.filename if siglip_pred else "",
                "siglip_top1_caption_correct": int(is_siglip_caption_correct),
                "truth_caption_in_candidates": int(has_truth_caption),
                "siglip_top_gap": f"{siglip_top_gap(candidates):.6f}",
                "rerank_applied": int(not (args.mock_reranker or args.final_score_mode == "siglip" or (args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args)))),
                "prediction_rerank_score": f"{float(pred.rerank_score if pred and pred.rerank_score is not None else 0.0):.6f}",
                "prediction_siglip_score": f"{float(pred.siglip_score if pred else 0.0):.6f}",
                "prediction_final_score": f"{float(ranked_scores.get(id(pred), 0.0) if pred else 0.0):.6f}",
                "ranked_candidates": "; ".join(
                    f"{candidate.filename}:{ranked_scores[id(candidate)]:.4f}"
                    for candidate in ranked_candidates
                ),
            }
        )

    total = len(target_rows)
    metrics = {
        "rows": total,
        "caption_accuracy": row_accuracy(correct_caption, total),
        "caption_correct": correct_caption,
        "class_accuracy": row_accuracy(correct_class, total),
        "class_correct": correct_class,
        "siglip_top1_caption_accuracy": row_accuracy(siglip_correct_caption, total),
        "siglip_top1_caption_correct": siglip_correct_caption,
        "truth_caption_in_rerank_candidates_rate": row_accuracy(truth_caption_in_candidates, total),
        "truth_caption_in_rerank_candidates": truth_caption_in_candidates,
        "rerank_pair_count": rerank_pair_count,
        "rerank_skipped_by_siglip_guard": rerank_skipped_by_siglip_guard,
    }
    timings["total_sec"] = time.perf_counter() - t0_total

    return {
        "schema_version": "orin_task2_caption_reranker_guarded_v1",
        "models": {
            "siglip_model": args.siglip_model,
            "siglip_pretrained": args.siglip_pretrained,
            "siglip_backend": args.siglip_backend,
            "siglip_onnx_path": str(args.siglip_onnx_path) if args.siglip_backend == "onnx" else "",
            "siglip_onnx_stage2_path": str(args.siglip_onnx_stage2_path) if args.siglip_backend == "onnx" else "",
            "siglip_onnx_provider": args.siglip_onnx_provider if args.siglip_backend == "onnx" else "",
            "siglip_precision": "fp16_compiled" if args.siglip_backend == "onnx" else args.torch_dtype,
            "reranker_size": args.reranker_size,
            "reranker_model": args.reranker_model,
            "reranker_backend": args.reranker_backend,
            "reranker_gguf": str(args.reranker_gguf) if args.reranker_backend == "mtmd" else "",
            "reranker_mmproj": str(args.reranker_mmproj) if args.reranker_backend == "mtmd" else "",
            "reranker_mtmd_compute": args.reranker_mtmd_compute if args.reranker_backend == "mtmd" else "",
            "reranker_mtmd_max_batch_candidates": args.reranker_mtmd_max_batch_candidates if args.reranker_backend == "mtmd" else None,
            "reranker_qairt_bundle": str(args.reranker_qairt_bundle) if args.reranker_backend == "qairt" and args.reranker_qairt_bundle else "",
            "reranker_qairt_contract": str(args.reranker_qairt_contract) if args.reranker_backend == "qairt" else "",
            "reranker_qairt_worker": str(args.reranker_qairt_executable) if args.reranker_backend == "qairt" else "",
            "reranker_qairt_backend_path": str(args.reranker_qairt_backend_path) if args.reranker_backend == "qairt" else "",
            "reranker_mode": reranker_mode,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
        },
        "settings": {
            "caption_text_mode": args.caption_text_mode,
            "siglip_image_size": args.siglip_image_size,
            "siglip_top_k": args.siglip_top_k,
            "rerank_top_k": args.rerank_top_k,
            "final_score_mode": args.final_score_mode,
            "rerank_weight": args.rerank_weight,
            "siglip_keep_gap": args.siglip_keep_gap,
            "reranker_mtmd_max_batch_candidates": args.reranker_mtmd_max_batch_candidates,
            "eval_first": args.eval_first,
            "eval_all": args.eval_all,
            "seed": args.seed,
        },
        "paths": {
            "image_dir": str(args.image_dir),
            "manifest": str(args.manifest),
            "images_list": str(args.images_list) if args.images_list is not None else "",
            "evaluation_json": str(args.evaluation_json),
            "caption_bank_json": str(caption_bank_json),
        },
        "caption_bank_count": len(caption_bank),
        "caption_cache": caption_cache,
        "image_cache": image_cache,
        "metrics": metrics,
        "timings_sec": timings,
        "traces": traces,
        "csv_rows": csv_rows,
    }


def resolve_output_paths(report: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    rows = report["metrics"]["rows"]
    if args.image is not None:
        stem = Path(str(args.image)).stem
        default_json = args.output_dir / f"{stem}_caption_alignment.json"
        default_csv = args.output_dir / f"{stem}_caption_alignment.csv"
    elif args.eval_all:
        default_json = args.output_dir / "eval_all_caption_alignment.json"
        default_csv = args.output_dir / "eval_all_caption_alignment.csv"
    elif args.eval_first:
        default_json = args.output_dir / f"eval_first_{rows}_caption_alignment.json"
        default_csv = args.output_dir / f"eval_first_{rows}_caption_alignment.csv"
    else:
        default_json = args.output_dir / f"eval_{rows}_caption_alignment.json"
        default_csv = args.output_dir / f"eval_{rows}_caption_alignment.csv"

    out_json = args.output_json or default_json
    out_csv = args.predictions_csv or default_csv
    return out_json, out_csv


def print_final_summary(report: dict[str, Any], out_json: Path, out_csv: Path) -> None:
    metrics = report["metrics"]
    if metrics.get("metric_available") is False:
        print(
            "Final metrics: inference_only_no_ground_truth "
            f"qwen_pairs={metrics['rerank_pair_count']} "
            f"siglip_guard_skips={metrics['rerank_skipped_by_siglip_guard']} "
            f"rows={metrics['rows']}",
            flush=True,
        )
    else:
        print(
            "Final metrics: "
            f"caption_accuracy={metrics['caption_accuracy']:.4f} "
            f"class_accuracy={metrics['class_accuracy']:.4f} "
            f"siglip_top1_caption={metrics['siglip_top1_caption_accuracy']:.4f} "
            f"truth_caption_in_candidates={metrics['truth_caption_in_rerank_candidates_rate']:.4f} "
            f"qwen_pairs={metrics['rerank_pair_count']} "
            f"siglip_guard_skips={metrics['rerank_skipped_by_siglip_guard']} "
            f"rows={metrics['rows']}",
            flush=True,
        )
    if "benchmark" in report:
        print_benchmark_summary(report["benchmark"])
    print(f"Planned Evaluation JSON: {out_json}", flush=True)
    print(f"Planned Predictions CSV: {out_csv}", flush=True)


def write_outputs(report: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json, out_csv = resolve_output_paths(report, args)
    json_payload = dict(report)
    csv_rows = json_payload.pop("csv_rows")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "truth_cat",
        "truth_caption",
        "prediction_filename",
        "prediction_cat",
        "prediction_caption",
        "caption_correct",
        "class_correct",
        "siglip_top1_filename",
        "siglip_top1_caption_correct",
        "truth_caption_in_candidates",
        "siglip_top_gap",
        "rerank_applied",
        "prediction_rerank_score",
        "prediction_siglip_score",
        "prediction_final_score",
        "siglip_image_and_recall_sec",
        "rerank_sec",
        "final_selection_sec",
        "total_image_sec",
        "ranked_candidates",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return out_json, out_csv


def main() -> None:
    args = parse_args()
    report = run_pipeline(args)
    out_json, out_csv = resolve_output_paths(report, args)
    print_final_summary(report, out_json, out_csv)
    out_json, out_csv = write_outputs(report, args)
    print(f"Evaluation JSON written: {out_json}")
    print(f"Predictions CSV written: {out_csv}")


if __name__ == "__main__":
    main()
