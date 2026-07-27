"""Shared profiling and acceptance helpers for the EVK acceleration work.

The helpers in this module deliberately have no accelerator dependencies so
reports can be inspected and compared on a development host.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SIGLIP_COMPONENTS = (
    "image_decode_preprocess_sec",
    "qnn_stage1_sec",
    "intermediate_transfer_sec",
    "qnn_stage2_sec",
    "normalization_sec",
    "cosine_recall_sec",
    "topk_selection_sec",
)

FIXED_IMAGE_SIZE = 384
FIXED_IMAGE_RESOLUTION = (FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE)


RERANKER_COMPONENTS_MS = (
    "image_resize_384_ms",
    "bitmap_load_ms",
    "tokenization_ms",
    "system_prefix_prefill_ms",
    "visual_encode_projector_ms",
    "image_kv_prefill_ms",
    "candidate_text_decode_ms",
    "rank_pooling_ms",
    "ipc_round_trip_ms",
    "ipc_overhead_ms",
)


def percentile(values: Iterable[float], q: float) -> float:
    samples = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(samples, q)) if samples.size else 0.0


def latency_summary(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "mean": float(np.mean(samples)) if samples else 0.0,
        "median": percentile(samples, 50.0),
        "p95": percentile(samples, 95.0),
        "minimum": min(samples, default=0.0),
        "maximum": max(samples, default=0.0),
    }


def component_summary(
    profiles: Iterable[dict[str, Any]],
    component_names: Iterable[str],
) -> dict[str, dict[str, float | int]]:
    rows = list(profiles)
    return {
        name: latency_summary(
            float(row.get("timings", row.get("timings_sec", row.get("timings_ms", {}))).get(name, 0.0))
            for row in rows
        )
        for name in component_names
    }


def recommended_context_capacity(
    maximum_sequence_tokens: int,
    *,
    headroom_tokens: int = 64,
    multiple: int = 256,
) -> int:
    if maximum_sequence_tokens < 0:
        raise ValueError("maximum_sequence_tokens cannot be negative")
    if headroom_tokens < 0 or multiple <= 0:
        raise ValueError("headroom_tokens must be non-negative and multiple must be positive")
    required = maximum_sequence_tokens + headroom_tokens
    return max(multiple, int(math.ceil(required / multiple)) * multiple)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path | None, *, hash_file: bool = True) -> dict[str, Any] | None:
    if path is None or not str(path):
        return None
    resolved = Path(path).expanduser().resolve()
    record: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
    }
    if resolved.is_file():
        stat = resolved.stat()
        record.update(
            {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(resolved) if hash_file else None,
            }
        )
    return record


def process_memory_snapshot(pid: int | None = None) -> dict[str, int]:
    """Return Linux process RSS/HWM values without adding a dependency."""
    target = int(pid if pid is not None else os.getpid())
    values = {"rss_bytes": 0, "peak_rss_bytes": 0}
    try:
        lines = Path(f"/proc/{target}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    keys = {"VmRSS:": "rss_bytes", "VmHWM:": "peak_rss_bytes"}
    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[0] in keys:
            values[keys[fields[0]]] = int(fields[1]) * 1024
    return values


def median_run_summary(run_summaries: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """Median of per-run mean/median/p95 values required by the protocol."""
    runs = list(run_summaries)
    return {
        "run_count": len(runs),
        "median_of_mean": percentile((float(row.get("mean", 0.0)) for row in runs), 50.0),
        "median_of_median": percentile((float(row.get("median", row.get("p50", 0.0))) for row in runs), 50.0),
        "median_of_p95": percentile((float(row.get("p95", 0.0)) for row in runs), 50.0),
    }


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Durably replace a text file without exposing a partially written report."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
        temp_name = None
        return destination
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Iterable[str],
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
        temp_name = None
        return destination
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def _command_output(command: list[str], timeout_sec: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _version_minor(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)", str(value))
    return f"{match.group(1)}.{match.group(2)}" if match else None


def collect_evk_stack_info(
    selected_qnn_backend_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect the two QNN user-space stacks that currently share HTP v73."""
    observed: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "fixed_image_resolution": list(FIXED_IMAGE_RESOLUTION),
        "onnxruntime": None,
        "onnxruntime_qnn": None,
        "onnxruntime_qnn_qairt": None,
        "geniex": None,
        "geniex_qairt": None,
        "geniex_llamacpp_hash": None,
        "libraries": {},
    }

    try:
        import onnxruntime as ort

        observed["onnxruntime"] = ort.__version__
    except Exception as exc:
        observed["onnxruntime_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import onnxruntime_qnn as qnn_ep
        from onnxruntime_qnn import build_and_package_info

        observed["onnxruntime_qnn"] = qnn_ep.__version__
        observed["onnxruntime_qnn_qairt"] = build_and_package_info.qnn_version
        for name, path in (
            ("ort_qnn_htp", qnn_ep.get_qnn_htp_path()),
            ("ort_qnn_ep", qnn_ep.get_library_path()),
        ):
            observed["libraries"][name] = artifact_record(path)
        qnn_root = Path(qnn_ep.get_qnn_htp_path()).parent
        for name in ("libQnnSystem.so", "libQnnHtpV73Skel.so"):
            observed["libraries"][f"ort_{name}"] = artifact_record(qnn_root / name)
    except Exception as exc:
        observed["onnxruntime_qnn_error"] = f"{type(exc).__name__}: {exc}"

    geniex = shutil.which("geniex")
    if geniex:
        version_text = _command_output([geniex, "--version"])
        observed["geniex_version_output"] = version_text
        match = re.search(r"GenieX CLI Version:\s*v?([^\s]+)", version_text)
        observed["geniex"] = match.group(1) if match else None
        match = re.search(r"QAIRT Runtime Version:\s*v?([^\s]+)", version_text)
        observed["geniex_qairt"] = match.group(1) if match else None
        match = re.search(r"LlamaCPP Runtime Hash:\s*([^\s]+)", version_text)
        observed["geniex_llamacpp_hash"] = match.group(1) if match else None
        geniex_root = Path(geniex).expanduser().resolve().parent.parent / "share/geniex"
        for relative in (
            "qairt/htp-files/libQnnHtp.so",
            "qairt/htp-files/libQnnSystem.so",
            "qairt/htp-files/libQnnHtpV73Skel.so",
            "llama_cpp/libggml-htp-v73.so",
        ):
            observed["libraries"][f"geniex_{Path(relative).name}"] = artifact_record(
                geniex_root / relative
            )
    else:
        observed["geniex_error"] = "geniex executable not found"

    selected_names = (
        "libQnnHtp.so",
        "libQnnSystem.so",
        "libQnnHtpV73Skel.so",
    )
    if selected_qnn_backend_path is not None:
        selected_backend = Path(selected_qnn_backend_path).expanduser().resolve()
        selected_root = selected_backend.parent
        observed["selected_qnn_backend_path"] = str(selected_backend)
        for name in selected_names:
            observed["libraries"][f"selected_{name}"] = artifact_record(
                selected_root / name
            )
    else:
        observed["selected_qnn_backend_path"] = None

    comparisons = {}
    for name in selected_names:
        selected = observed["libraries"].get(f"selected_{name}")
        geniex_library = observed["libraries"].get(f"geniex_{name}")
        comparisons[name] = {
            "selected": selected,
            "geniex": geniex_library,
            "matches": bool(
                selected
                and geniex_library
                and selected.get("sha256") == geniex_library.get("sha256")
            ),
        }
    observed["selected_qnn_stack_comparison"] = comparisons
    observed["selected_qnn_stack_matches_geniex"] = all(
        item["matches"] for item in comparisons.values()
    )

    qairt_versions = [
        value
        for value in (
            observed.get("onnxruntime_qnn_qairt"),
            observed.get("geniex_qairt"),
        )
        if value
    ]
    qairt_minors = sorted(
        {minor for value in qairt_versions if (minor := _version_minor(str(value))) is not None}
    )
    observed["qairt_runtime_versions"] = qairt_versions
    observed["qairt_runtime_minors"] = qairt_minors
    observed["single_qairt_minor"] = len(qairt_minors) == 1
    return observed


def evaluate_evk_stack(
    observed: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    expected = manifest.get("expected", {})
    mismatches = []
    for key, expected_value in expected.items():
        actual = observed.get(key)
        if expected_value is not None and str(actual) != str(expected_value):
            mismatches.append({"component": key, "expected": expected_value, "observed": actual})

    policy = manifest.get("policy", {})
    violations = []
    for component in policy.get("required_components", []):
        if not observed.get(component):
            violations.append(
                {"policy": "required_component", "component": component, "observed": None}
            )
    if policy.get("require_single_qairt_minor") and not observed.get("single_qairt_minor"):
        violations.append(
            {
                "policy": "require_single_qairt_minor",
                "observed": observed.get("qairt_runtime_minors", []),
            }
        )
    expected_minor = policy.get("expected_qairt_minor")
    if expected_minor and observed.get("qairt_runtime_minors") != [str(expected_minor)]:
        violations.append(
            {
                "policy": "expected_qairt_minor",
                "expected": str(expected_minor),
                "observed": observed.get("qairt_runtime_minors", []),
            }
        )
    if (
        policy.get("require_selected_qnn_stack_matches_geniex")
        and not observed.get("selected_qnn_stack_matches_geniex")
    ):
        violations.append(
            {
                "policy": "require_selected_qnn_stack_matches_geniex",
                "observed": observed.get("selected_qnn_stack_comparison", {}),
            }
        )
    required_resolution = policy.get("fixed_image_resolution")
    if required_resolution and list(required_resolution) != list(
        observed.get("fixed_image_resolution", [])
    ):
        violations.append(
            {
                "policy": "fixed_image_resolution",
                "expected": required_resolution,
                "observed": observed.get("fixed_image_resolution"),
            }
        )
    return {
        "compatible": not mismatches and not violations,
        "manifest_status": manifest.get("status", "unknown"),
        "version_mismatches": mismatches,
        "policy_violations": violations,
    }


def load_and_check_evk_stack(
    manifest_path: str | Path,
    *,
    selected_qnn_backend_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    observed = collect_evk_stack_info(
        selected_qnn_backend_path=selected_qnn_backend_path
    )
    return {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "observed": observed,
        "assessment": evaluate_evk_stack(observed, manifest),
    }


def git_revision(root: str | Path) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve()
    revision = _command_output(["git", "-C", str(directory), "rev-parse", "HEAD"])
    dirty = bool(_command_output(["git", "-C", str(directory), "status", "--porcelain"]))
    return {"revision": revision or None, "dirty": dirty}


def base_run_config(root: str | Path) -> dict[str, Any]:
    return {
        "schema_version": "dishcovery_run_config_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "source": git_revision(root),
        "fixed_image_resolution": list(FIXED_IMAGE_RESOLUTION),
    }
