#!/usr/bin/env python3
"""Run a byte-identical image/prompt probe against GenieX and Edge-LLM.

The probe deliberately freezes the SigLIP output. ``prepare`` copies prompts and
candidate lists from an existing Task 1 report and creates one canonical,
lossless 512x512 PNG per image. ``run`` sends those exact PNG files and prompts
directly to a selected VLM backend; there is no visual-gap skip and no
intermediate JPEG conversion.

The prepared directory is portable. Copy the whole directory to the other
device instead of regenerating its images there.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from PIL import Image, __version__ as PILLOW_VERSION


SCHEMA_VERSION = "dishcovery_same_pixel_qwen_manifest_v1"
RESULT_SCHEMA_VERSION = "dishcovery_same_pixel_qwen_result_v1"
COMPARISON_SCHEMA_VERSION = "dishcovery_same_pixel_qwen_comparison_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def image_name_from_trace(trace: dict[str, Any]) -> str:
    raw = str(trace.get("image") or "").strip()
    return Path(raw).name if raw else ""


def traces_by_image(report: dict[str, Any], source: Path) -> dict[str, dict[str, Any]]:
    traces = report.get("traces")
    if not isinstance(traces, list):
        raise TypeError(f"{source} has no traces array")
    indexed: dict[str, dict[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        name = image_name_from_trace(trace)
        if not name:
            continue
        if name in indexed:
            raise ValueError(f"Duplicate trace for {name} in {source}")
        indexed[name] = trace
    if not indexed:
        raise ValueError(f"No image traces found in {source}")
    return indexed


def qwen_was_run(trace: dict[str, Any]) -> bool:
    qwen = trace.get("qwen") if isinstance(trace.get("qwen"), dict) else {}
    skip = trace.get("skip_vlm") if isinstance(trace.get("skip_vlm"), dict) else {}
    return not bool(qwen.get("skipped") or skip.get("skipped"))


def trace_prompt(trace: dict[str, Any], image_name: str) -> str:
    qwen = trace.get("qwen") if isinstance(trace.get("qwen"), dict) else {}
    prompt = qwen.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Trace for {image_name} has no Qwen prompt")
    return prompt


def trace_candidates(trace: dict[str, Any], image_name: str) -> list[dict[str, Any]]:
    raw_candidates = trace.get("siglip2_top_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"Trace for {image_name} has no SigLIP candidate list")
    candidates: list[dict[str, Any]] = []
    for fallback_rank, raw in enumerate(raw_candidates, start=1):
        if not isinstance(raw, dict):
            raise TypeError(f"Invalid candidate {fallback_rank} for {image_name}")
        label = str(raw.get("label") or "").strip()
        if not label:
            raise ValueError(f"Candidate {fallback_rank} for {image_name} has no label")
        candidates.append(
            {
                "rank": int(raw.get("rank") or fallback_rank),
                "label_id": int(raw["label_id"]) if raw.get("label_id") is not None else None,
                "label": label,
                "visual_score": (
                    float(raw["visual_score"]) if raw.get("visual_score") is not None else None
                ),
            }
        )
    ranks = [int(item["rank"]) for item in candidates]
    if ranks != list(range(1, len(candidates) + 1)):
        raise ValueError(f"Candidate ranks for {image_name} are not contiguous and 1-based: {ranks}")
    return candidates


def read_truth_csv(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    truth: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_name = Path(str(row.get("image") or "")).name
            if not image_name:
                continue
            truth[image_name] = [
                label.strip()
                for label in str(row.get("truth") or "").split("|")
                if label.strip()
            ]
    return truth


def evenly_spaced(values: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if count >= len(values):
        return list(values)
    if count == 1:
        return [values[0]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]


def select_images(
    reference: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, Any]] | None,
    *,
    selection: str,
    limit: int,
    includes: list[str],
) -> list[str]:
    names = sorted(reference)
    if selection in {"reference-qwen", "common-qwen"}:
        names = [name for name in names if qwen_was_run(reference[name])]
    if selection == "common-qwen":
        if paired is None:
            raise ValueError("--selection common-qwen requires --paired-report")
        names = [name for name in names if name in paired and qwen_was_run(paired[name])]

    include_names = [Path(value).name for value in includes]
    missing_includes = [name for name in include_names if name not in names]
    if missing_includes:
        raise ValueError(
            "Explicitly included images do not satisfy the selection filter or are absent: "
            + ", ".join(missing_includes)
        )
    include_names = list(dict.fromkeys(include_names))
    if limit > 0 and len(include_names) > limit:
        raise ValueError(f"--limit {limit} is smaller than the explicit include count")

    remaining = [name for name in names if name not in set(include_names)]
    remaining_limit = 0 if limit <= 0 else max(0, limit - len(include_names))
    selected = include_names + evenly_spaced(remaining, remaining_limit)
    if limit <= 0:
        selected = include_names + remaining
    return selected


def source_image_path(image_dir: Path, image_name: str) -> Path:
    direct = image_dir / image_name
    if direct.exists():
        return direct
    stem = Path(image_name).stem
    matches = sorted(
        path for path in image_dir.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Source image not found: {direct}")
    raise ValueError(f"Multiple source images match {image_name}: {matches}")


def canonicalize_image(source: Path, destination: Path, side: int) -> dict[str, Any]:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        original_size = [image.width, image.height]
        scale = min(side / image.width, side / image.height)
        resized_size = [
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ]
        resized = image.resize(tuple(resized_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    offset = [(side - resized.width) // 2, (side - resized.height) // 2]
    canvas.paste(resized, tuple(offset))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Lossless, metadata-free output. The prepared bytes are copied to the other
    # device; they must never be regenerated independently.
    canvas.save(destination, format="PNG", compress_level=9, optimize=False)
    return {
        "source_size": original_size,
        "resized_size": resized_size,
        "canvas_size": [side, side],
        "paste_offset": offset,
        "mode": "RGB",
        "sha256": sha256_file(destination),
        "byte_count": destination.stat().st_size,
    }


def prepare(args: argparse.Namespace) -> None:
    reference_report = load_json(args.reference_report)
    reference = traces_by_image(reference_report, args.reference_report)
    paired = None
    if args.paired_report is not None:
        paired = traces_by_image(load_json(args.paired_report), args.paired_report)
    truth = read_truth_csv(args.truth_csv)
    selected = select_images(
        reference,
        paired,
        selection=args.selection,
        limit=args.limit,
        includes=args.include_image,
    )
    if not selected:
        raise ValueError("The selection produced no images")

    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; choose another directory or pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_destination = output_dir / Path(__file__).name
    if Path(__file__).resolve() != runner_destination.resolve():
        shutil.copy2(Path(__file__).resolve(), runner_destination)

    items: list[dict[str, Any]] = []
    for index, image_name in enumerate(selected, start=1):
        trace = reference[image_name]
        prompt = trace_prompt(trace, image_name)
        source = source_image_path(args.image_dir, image_name)
        canonical_name = f"{Path(image_name).stem}.png"
        destination = images_dir / canonical_name
        image_record = canonicalize_image(source, destination, args.side)
        qwen = trace.get("qwen") if isinstance(trace.get("qwen"), dict) else {}
        paired_trace = paired.get(image_name) if paired is not None else None
        items.append(
            {
                "index": index,
                "image_name": image_name,
                "source_image_sha256": sha256_file(source),
                "canonical_path": str(Path("images") / canonical_name),
                "canonical": image_record,
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "candidates": trace_candidates(trace, image_name),
                "ground_truth": truth.get(image_name, []),
                "historical": {
                    "reference_qwen_run": qwen_was_run(trace),
                    "reference_qwen_timed_out": bool(qwen.get("timed_out")),
                    "paired_qwen_run": (
                        qwen_was_run(paired_trace) if isinstance(paired_trace, dict) else None
                    ),
                },
            }
        )
        print(f"[{index}/{len(selected)}] prepared {canonical_name} {image_record['sha256']}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "test_contract": {
            "qwen_forced_for_every_item": True,
            "siglip_candidates_frozen": True,
            "prompt_frozen": True,
            "canonical_format": "PNG",
            "canonical_mode": "RGB",
            "canonical_size": [args.side, args.side],
            "resize": "Pillow LANCZOS, aspect ratio preserved",
            "padding": "centered white RGB(255,255,255)",
            "intermediate_jpeg": False,
            "pillow_version_at_prepare": PILLOW_VERSION,
        },
        "selection": {
            "mode": args.selection,
            "limit": args.limit,
            "explicit_includes": [Path(value).name for value in args.include_image],
            "selected_count": len(items),
        },
        "sources": {
            "reference_report": str(args.reference_report.resolve()),
            "reference_report_sha256": sha256_file(args.reference_report),
            "paired_report": str(args.paired_report.resolve()) if args.paired_report else None,
            "paired_report_sha256": (
                sha256_file(args.paired_report) if args.paired_report else None
            ),
            "truth_csv": str(args.truth_csv.resolve()) if args.truth_csv else None,
            "truth_csv_sha256": sha256_file(args.truth_csv) if args.truth_csv else None,
            "image_dir": str(args.image_dir.resolve()),
        },
        "items": items,
    }
    manifest_path = output_dir / "same_pixel_manifest.json"
    write_json(manifest_path, manifest)

    checksummed = [manifest_path, runner_destination]
    checksummed.extend(output_dir / str(item["canonical_path"]) for item in items)
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir)}" for path in checksummed
    ]
    (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Prepared {len(items)} forced-Qwen cases in {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("No model inference was run.")


def manifest_items(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{manifest_path} has no test items")
    typed_items = [item for item in items if isinstance(item, dict)]
    if len(typed_items) != len(items):
        raise TypeError(f"{manifest_path} contains a non-object test item")
    return manifest, typed_items


def canonical_path(manifest_path: Path, item: dict[str, Any]) -> Path:
    raw = Path(str(item.get("canonical_path") or ""))
    if raw.is_absolute():
        raise ValueError(f"Canonical paths must be relative for portability: {raw}")
    path = (manifest_path.resolve().parent / raw).resolve()
    try:
        path.relative_to(manifest_path.resolve().parent)
    except ValueError as exc:
        raise ValueError(f"Canonical path escapes the prepared directory: {raw}") from exc
    return path


def verify_item(manifest_path: Path, item: dict[str, Any]) -> dict[str, Any]:
    path = canonical_path(manifest_path, item)
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical image: {path}")
    expected = str((item.get("canonical") or {}).get("sha256") or "")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"Canonical image hash mismatch for {item.get('image_name')}: "
            f"expected {expected}, observed {observed}"
        )
    prompt = str(item.get("prompt") or "")
    prompt_hash = sha256_bytes(prompt.encode("utf-8"))
    if prompt_hash != item.get("prompt_sha256"):
        raise ValueError(f"Prompt hash mismatch for {item.get('image_name')}")
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        image.verify()
    expected_size = list((item.get("canonical") or {}).get("canvas_size") or [])
    if [width, height] != expected_size or mode != "RGB":
        raise ValueError(
            f"Unexpected canonical image properties for {item.get('image_name')}: "
            f"{width}x{height} {mode}, expected {expected_size} RGB"
        )
    return {
        "path": str(path),
        "sha256": observed,
        "size": [width, height],
        "mode": mode,
    }


def verify(args: argparse.Namespace) -> None:
    _, items = manifest_items(args.manifest)
    for index, item in enumerate(items, start=1):
        record = verify_item(args.manifest, item)
        print(
            f"[{index}/{len(items)}] verified {item.get('image_name')} "
            f"{record['sha256']}"
        )
    print(f"Verified {len(items)} canonical PNGs and prompts; no inference was run.")


def extract_json_object(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stripped[start : index + 1])
                except Exception:
                    return None
    return None


def parse_rank_array(
    payload: dict[str, Any],
    key: str,
    candidate_count: int,
    warnings: list[str],
) -> list[int]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        warnings.append(f"{key}_not_array")
        return []
    parsed: list[int] = []
    seen: set[int] = set()
    for value in raw:
        if isinstance(value, bool):
            warnings.append(f"{key}_boolean")
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if not isinstance(value, int):
            warnings.append(f"{key}_non_integer")
            continue
        if value < 1 or value > candidate_count:
            warnings.append(f"{key}_out_of_range:{value}")
            continue
        if value in seen:
            warnings.append(f"{key}_duplicate:{value}")
            continue
        seen.add(value)
        parsed.append(value)
    return parsed


def parse_classification(raw_text: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    payload = extract_json_object(raw_text)
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return {
            "valid_json": False,
            "valid_schema": False,
            "present": [],
            "possible": [],
            "nonzero": [],
            "present_labels": [],
            "possible_labels": [],
            "warnings": ["no_json_object"],
        }
    if set(payload) != {"present", "possible"}:
        warnings.append("keys_not_exactly_present_possible")
    present = parse_rank_array(payload, "present", len(candidates), warnings)
    possible = parse_rank_array(payload, "possible", len(candidates), warnings)
    overlap = sorted(set(present) & set(possible))
    if overlap:
        warnings.append("present_possible_overlap:" + ",".join(map(str, overlap)))
        possible = [rank for rank in possible if rank not in set(present)]
    labels = {int(item["rank"]): str(item["label"]) for item in candidates}
    return {
        "valid_json": True,
        "valid_schema": not warnings,
        "present": present,
        "possible": possible,
        "nonzero": sorted(set(present) | set(possible)),
        "present_labels": [labels[rank] for rank in present],
        "possible_labels": [labels[rank] for rank in possible],
        "warnings": warnings,
    }


class GenieXBackend:
    name = "geniex"

    def __init__(self, args: argparse.Namespace) -> None:
        self.url = str(args.geniex_url)
        self.model = str(args.geniex_model)
        self.max_new_tokens = int(args.max_new_tokens)
        self.timeout_sec = float(args.timeout_sec)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "url": self.url,
            "model": self.model,
            "request_generation": {
                "max_completion_tokens": self.max_new_tokens,
                "max_tokens": self.max_new_tokens,
                "temperature": -1.0,
                "stream": False,
            },
            "external_image_preprocessing": "disabled; canonical PNG path sent directly",
        }

    def generate(self, image_path: Path, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": str(image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_completion_tokens": self.max_new_tokens,
            "max_tokens": self.max_new_tokens,
            "temperature": -1.0,
            "stream": False,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = self.timeout_sec if self.timeout_sec > 0 else 900.0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GenieX HTTP {exc.code}: {detail}") from exc
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"GenieX response has no choices: {data}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise RuntimeError(f"GenieX response has no message: {data}")
        return str(message.get("content") or "")

    def close(self) -> None:
        return


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("controlled_probe_edgellm_qwen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Edge-LLM adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EdgeLLMBackend:
    name = "edgellm"

    def __init__(self, args: argparse.Namespace) -> None:
        adapter_path = args.edgellm_adapter.expanduser().resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"Edge-LLM adapter not found: {adapter_path}")
        module = load_module(adapter_path)
        config_args = argparse.Namespace(
            edgellm_root=args.edgellm_root,
            edgellm_workspace=args.edgellm_workspace,
            edgellm_model_name=args.edgellm_model_name,
            edgellm_llm_engine_profile=args.edgellm_llm_engine_profile,
            edgellm_llm_engine_dir=args.edgellm_llm_engine_dir,
            edgellm_visual_engine_dir=args.edgellm_visual_engine_dir,
            edgellm_binary=args.edgellm_binary,
            edgellm_persistent_binary=args.edgellm_persistent_binary,
            edgellm_plugin_path=args.edgellm_plugin_path,
            edgellm_runtime_mode=args.edgellm_runtime_mode,
            edgellm_timeout_sec=args.timeout_sec,
            edgellm_startup_timeout_sec=args.edgellm_startup_timeout_sec,
            edgellm_keep_io=args.edgellm_keep_io,
            edgellm_dump_profile=False,
            edgellm_warmup=args.edgellm_warmup,
            edgellm_temperature=0.0,
            edgellm_top_p=1.0,
            edgellm_top_k=1,
            edgellm_enable_thinking=False,
        )
        self.module = module
        self.config = module.config_from_args(config_args)
        self.scorer = module.EdgeLLMQwenScorer(
            self.config,
            max_new_tokens=int(args.max_new_tokens),
        )
        self.max_new_tokens = int(args.max_new_tokens)
        self.max_pixels = int(args.edge_max_pixels)
        self.adapter_path = adapter_path

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "adapter": str(self.adapter_path),
            "adapter_sha256": sha256_file(self.adapter_path),
            "model": self.config.model_name,
            "runtime_mode": self.config.runtime_mode,
            "llm_engine_dir": str(self.config.llm_engine_dir),
            "visual_engine_dir": str(self.config.visual_engine_dir),
            "generation": {
                "max_generate_length": self.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "enable_thinking": self.config.enable_thinking,
            },
            "external_image_preprocessing": "disabled; canonical PNG path sent directly",
            "request_max_pixels": self.max_pixels,
        }

    def generate(self, image_path: Path, prompt: str) -> str:
        return str(
            self.scorer.score(
                image_path,
                prompt,
                max_new_tokens=self.max_new_tokens,
                min_pixels=None,
                max_pixels=self.max_pixels if self.max_pixels > 0 else None,
            )
        )

    def close(self) -> None:
        close = getattr(self.scorer, "close", None)
        if callable(close):
            close()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def metrics_from_sets(predictions: Iterable[set[str]], truths: Iterable[set[str]]) -> dict[str, Any]:
    tp = fp = fn = 0
    for prediction, truth in zip(predictions, truths):
        tp += len(prediction & truth)
        fp += len(prediction - truth)
        fn += len(truth - prediction)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if row.get("status") == "ok"]
    parsed = [row.get("parsed") or {} for row in ok]
    nonzero_counts = [len(row.get("nonzero") or []) for row in parsed]
    latencies = [float(row.get("latency_sec") or 0.0) for row in ok]
    present_predictions: list[set[str]] = []
    nonzero_predictions: list[set[str]] = []
    candidate_truths: list[set[str]] = []
    for row in ok:
        parsed_row = row.get("parsed") or {}
        present_predictions.append(set(map(str, parsed_row.get("present_labels") or [])))
        nonzero_predictions.append(
            set(map(str, parsed_row.get("present_labels") or []))
            | set(map(str, parsed_row.get("possible_labels") or []))
        )
        candidate_labels = set(map(str, row.get("candidate_labels") or []))
        candidate_truths.append(set(map(str, row.get("ground_truth") or [])) & candidate_labels)
    return {
        "rows": len(results),
        "ok_rows": len(ok),
        "error_rows": len(results) - len(ok),
        "valid_json_rows": sum(bool(row.get("valid_json")) for row in parsed),
        "valid_schema_rows": sum(bool(row.get("valid_schema")) for row in parsed),
        "empty_nonzero_rows": sum(count == 0 for count in nonzero_counts),
        "nonzero_19_or_20_rows": sum(count >= 19 for count in nonzero_counts),
        "mean_present": (
            statistics.fmean(len(row.get("present") or []) for row in parsed) if parsed else None
        ),
        "mean_possible": (
            statistics.fmean(len(row.get("possible") or []) for row in parsed) if parsed else None
        ),
        "mean_nonzero": statistics.fmean(nonzero_counts) if nonzero_counts else None,
        "median_nonzero": statistics.median(nonzero_counts) if nonzero_counts else None,
        "p95_nonzero": percentile([float(value) for value in nonzero_counts], 0.95),
        "mean_latency_sec": statistics.fmean(latencies) if latencies else None,
        "candidate_ground_truth_coverage_rows": sum(bool(value) for value in candidate_truths),
        "present_only_candidate_metrics": metrics_from_sets(
            present_predictions, candidate_truths
        ),
        "present_plus_possible_candidate_metrics": metrics_from_sets(
            nonzero_predictions, candidate_truths
        ),
    }


def result_shell(
    args: argparse.Namespace,
    backend: GenieXBackend | EdgeLLMBackend,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "forced_qwen": True,
        "backend": backend.describe(),
        "results": [],
        "summary": {},
    }


def run_probe(args: argparse.Namespace) -> None:
    _, items = manifest_items(args.manifest)
    for item in items:
        verify_item(args.manifest, item)
    manifest_hash = sha256_file(args.manifest)

    backend: GenieXBackend | EdgeLLMBackend
    if args.backend == "geniex":
        backend = GenieXBackend(args)
    else:
        backend = EdgeLLMBackend(args)

    output = args.output.resolve()
    report = result_shell(args, backend, manifest_hash)
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and output.exists():
        previous = load_json(output)
        if previous.get("manifest_sha256") != manifest_hash:
            raise ValueError("Cannot resume: output was created from a different manifest")
        if (previous.get("backend") or {}).get("backend") != args.backend:
            raise ValueError("Cannot resume: output uses a different backend")
        completed = {
            str(row.get("image_name")): row
            for row in previous.get("results") or []
            if isinstance(row, dict) and row.get("status") == "ok"
        }
        report["results"] = list(completed.values())

    try:
        for index, item in enumerate(items, start=1):
            image_name = str(item.get("image_name"))
            if image_name in completed:
                print(f"[{index}/{len(items)}] resume-skip {image_name}")
                continue
            image_path = canonical_path(args.manifest, item)
            before_hash = sha256_file(image_path)
            started = time.perf_counter()
            raw_text = ""
            error: dict[str, str] | None = None
            try:
                # Qwen is intentionally called for every manifest item. There is
                # no SigLIP gap check or VLM skip branch in this runner.
                raw_text = backend.generate(image_path, str(item.get("prompt") or ""))
                status = "ok"
            except Exception as exc:
                status = "error"
                error = {"type": type(exc).__name__, "message": str(exc)}
                if args.fail_fast:
                    raise
            latency = time.perf_counter() - started
            after_hash = sha256_file(image_path)
            if after_hash != before_hash:
                raise RuntimeError(f"Backend modified canonical input bytes: {image_path}")
            candidates = list(item.get("candidates") or [])
            parsed = parse_classification(raw_text, candidates) if status == "ok" else None
            row = {
                "index": int(item.get("index") or index),
                "image_name": image_name,
                "canonical_path": str(image_path),
                "canonical_sha256_before": before_hash,
                "canonical_sha256_after": after_hash,
                "prompt_sha256": item.get("prompt_sha256"),
                "candidate_labels": [str(value.get("label")) for value in candidates],
                "ground_truth": list(item.get("ground_truth") or []),
                "status": status,
                "latency_sec": latency,
                "raw_text": raw_text,
                "parsed": parsed,
                "error": error,
            }
            completed[image_name] = row
            report["results"] = [
                completed[name]
                for name in [str(value.get("image_name")) for value in items]
                if name in completed
            ]
            report["summary"] = summarize_results(report["results"])
            write_json(output, report)
            nonzero = len((parsed or {}).get("nonzero") or [])
            print(
                f"[{index}/{len(items)}] {image_name} status={status} "
                f"nonzero={nonzero} time={latency:.3f}s"
            )
    finally:
        backend.close()

    report["summary"] = summarize_results(report["results"])
    write_json(output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Result: {output}")


def result_rows(report: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    if report.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported result schema in {path}: {report.get('schema_version')}")
    rows = report.get("results")
    if not isinstance(rows, list):
        raise TypeError(f"{path} has no results array")
    return {
        str(row.get("image_name")): row
        for row in rows
        if isinstance(row, dict) and row.get("image_name")
    }


def compare(args: argparse.Namespace) -> None:
    _, items = manifest_items(args.manifest)
    manifest_hash = sha256_file(args.manifest)
    left_report = load_json(args.left)
    right_report = load_json(args.right)
    for name, report, path in (
        ("left", left_report, args.left),
        ("right", right_report, args.right),
    ):
        if report.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"{name} result {path} does not match the supplied manifest")
    left = result_rows(left_report, args.left)
    right = result_rows(right_report, args.right)

    paired: list[dict[str, Any]] = []
    for item in items:
        image_name = str(item.get("image_name"))
        left_row = left.get(image_name)
        right_row = right.get(image_name)
        if left_row is None or right_row is None:
            continue
        left_parsed = left_row.get("parsed") if isinstance(left_row.get("parsed"), dict) else {}
        right_parsed = (
            right_row.get("parsed") if isinstance(right_row.get("parsed"), dict) else {}
        )
        paired.append(
            {
                "image_name": image_name,
                "left_status": left_row.get("status"),
                "right_status": right_row.get("status"),
                "same_raw_text": left_row.get("raw_text") == right_row.get("raw_text"),
                "same_present": left_parsed.get("present") == right_parsed.get("present"),
                "same_possible": left_parsed.get("possible") == right_parsed.get("possible"),
                "left_present_count": len(left_parsed.get("present") or []),
                "right_present_count": len(right_parsed.get("present") or []),
                "left_nonzero_count": len(left_parsed.get("nonzero") or []),
                "right_nonzero_count": len(right_parsed.get("nonzero") or []),
                "nonzero_delta_right_minus_left": (
                    len(right_parsed.get("nonzero") or [])
                    - len(left_parsed.get("nonzero") or [])
                ),
            }
        )

    paired_ok = [
        row
        for row in paired
        if row["left_status"] == "ok" and row["right_status"] == "ok"
    ]
    deltas = [float(row["nonzero_delta_right_minus_left"]) for row in paired_ok]
    comparison = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "manifest_sha256": manifest_hash,
        "left": {
            "path": str(args.left.resolve()),
            "backend": left_report.get("backend"),
            "summary": left_report.get("summary"),
        },
        "right": {
            "path": str(args.right.resolve()),
            "backend": right_report.get("backend"),
            "summary": right_report.get("summary"),
        },
        "paired_summary": {
            "manifest_rows": len(items),
            "paired_rows": len(paired),
            "paired_ok_rows": len(paired_ok),
            "same_raw_text_rows": sum(bool(row["same_raw_text"]) for row in paired_ok),
            "same_present_rows": sum(bool(row["same_present"]) for row in paired_ok),
            "same_possible_rows": sum(bool(row["same_possible"]) for row in paired_ok),
            "mean_nonzero_delta_right_minus_left": (
                statistics.fmean(deltas) if deltas else None
            ),
            "median_nonzero_delta_right_minus_left": (
                statistics.median(deltas) if deltas else None
            ),
        },
        "paired": paired,
    }
    write_json(args.output, comparison)
    print(json.dumps(comparison["paired_summary"], indent=2, sort_keys=True))
    print(f"Comparison: {args.output.resolve()}")


def add_edge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--edgellm-adapter",
        type=Path,
        default=Path("/home/ubuntu/Desktop/dishcovery_orin/code/edgellm_qwen.py"),
    )
    parser.add_argument(
        "--edgellm-root",
        type=Path,
        default=Path("/home/ubuntu/Desktop/dishcovery_orin/TensorRT-Edge-LLM"),
    )
    parser.add_argument(
        "--edgellm-workspace",
        type=Path,
        default=Path("/home/danilopau/tensorrt-edgellm-workspace"),
    )
    parser.add_argument("--edgellm-model-name", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--edgellm-llm-engine-profile", default="default")
    parser.add_argument("--edgellm-llm-engine-dir", type=Path, default=None)
    parser.add_argument("--edgellm-visual-engine-dir", type=Path, default=None)
    parser.add_argument("--edgellm-binary", type=Path, default=None)
    parser.add_argument("--edgellm-persistent-binary", type=Path, default=None)
    parser.add_argument("--edgellm-plugin-path", type=Path, default=None)
    parser.add_argument(
        "--edgellm-runtime-mode",
        choices=("subprocess", "persistent"),
        default="persistent",
    )
    parser.add_argument("--edgellm-startup-timeout-sec", type=float, default=240.0)
    parser.add_argument("--edgellm-keep-io", action="store_true")
    parser.add_argument("--edgellm-warmup", type=int, default=0)
    parser.add_argument(
        "--edge-max-pixels",
        type=int,
        default=512 * 512,
        help="Per-request Edge-LLM max_pixels metadata. The canonical PNG remains unchanged.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, verify, run, and compare a byte-identical 512x512 PNG Qwen probe. "
            "The run subcommand always executes Qwen for every manifest item."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Create the portable canonical-PNG/prompt bundle without running inference.",
    )
    prepare_parser.add_argument("--reference-report", type=Path, required=True)
    prepare_parser.add_argument("--paired-report", type=Path, default=None)
    prepare_parser.add_argument("--truth-csv", type=Path, default=None)
    prepare_parser.add_argument("--image-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--selection",
        choices=("all", "reference-qwen", "common-qwen"),
        default="common-qwen",
    )
    prepare_parser.add_argument("--limit", type=int, default=100)
    prepare_parser.add_argument("--include-image", action="append", default=[])
    prepare_parser.add_argument("--side", type=int, default=512)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify canonical hashes, dimensions, and prompt hashes without inference.",
    )
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.set_defaults(func=verify)

    run_parser = subparsers.add_parser(
        "run",
        help="Force one Qwen request for every manifest row; no SigLIP skip is possible.",
    )
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--backend", choices=("geniex", "edgellm"), required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--max-new-tokens", type=int, default=96)
    run_parser.add_argument("--timeout-sec", type=float, default=60.0)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument(
        "--geniex-url",
        default="http://127.0.0.1:18181/v1/chat/completions",
    )
    run_parser.add_argument("--geniex-model", default="local/qwen3vl-4b-qairt-w4a16")
    add_edge_args(run_parser)
    run_parser.set_defaults(func=run_probe)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Create a paired selectivity/accuracy comparison from two result files.",
    )
    compare_parser.add_argument("--manifest", type=Path, required=True)
    compare_parser.add_argument("--left", type=Path, required=True)
    compare_parser.add_argument("--right", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(func=compare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
