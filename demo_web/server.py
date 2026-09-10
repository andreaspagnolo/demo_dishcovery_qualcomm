#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from backend.command_parser import CommandParseError, DemoCommand, TaskName, parse_command
from backend.history import ResultHistory
from backend.nutrition import NutritionStore, parse_date
from backend.qairt_speech import (
    DEFAULT_SPEECH_MODEL_ROOT,
    piper_bundle_path,
    whisper_bundle_path,
)
from backend.stt import DEFAULT_STT_BACKEND, DEFAULT_WHISPER_MODEL, SpeechToText
from backend.tts import TextToSpeech
from backend.task_router import TaskRouter, TaskRouterConfig


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "backend" / "web_static"
MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_AUDIO_BYTES = 4 * 1024 * 1024
DEFAULT_PIPER_MODEL = Path("external_assets/models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx")
DEFAULT_PIPER_CONFIG = Path("external_assets/models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json")
DEFAULT_HISTORY_ROOT = Path("demo_runs/web")
DEFAULT_NUTRITION_ROOT = DEFAULT_HISTORY_ROOT / "nutrition"
# The curated images are the browser demo's default gallery. Leave the list
# unset so ``sample_images`` enumerates this directory on every startup.
DEFAULT_SAMPLE_IMAGE_DIR = Path("external_assets/images/demo")
DEFAULT_SAMPLE_IMAGES_LIST: Path | None = None
DEFAULT_SAMPLE_IMAGE_COUNT = 80
DEFAULT_PRELOAD_BACKENDS = ["task1"]
MAX_JOBS = 30
DIAGNOSTICS_POWER_INTERVAL_MS = 200
WEB_VOICE_SECONDS = 6.0
WEB_STT_SAMPLE_RATE = 16_000
TASKS: set[str] = {"task1", "task2", "calories", "both"}
TASK_COMMAND_TEXT = {
    "task1": "find ingredients",
    "task2": "describe the dish",
    "calories": "estimate calories",
    "both": "execute both",
}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_length = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length") from exc
    if length <= 0:
        return {}
    if length > MAX_IMAGE_BYTES * 2:
        raise ValueError("Request body is too large")
    body = handler.rfile.read(length)
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def extract_hf_env_value(line: str, key: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    name, value = stripped.split("=", 1)
    if name.strip() != key:
        return None
    return value.strip().strip("'").strip('"') or None


def read_token_file(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        for raw_line in text.splitlines():
            value = extract_hf_env_value(raw_line, key)
            if value:
                return value
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            return line.strip().strip("'").strip('"') or None
    return None


def read_hf_token() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    for path in (Path(".env"), Path(".hf_token"), Path.home() / ".cache/huggingface/token"):
        token = read_token_file(path.expanduser())
        if token:
            return token
    return None


def format_latency(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), 2)


def human_list(text: str) -> str:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) <= 1:
        return text
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def spoken_answer(result: dict[str, Any]) -> str:
    task = result.get("task")
    answer = str(result.get("answer") or "").strip()
    if task == "both":
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        task1 = details.get("task1") if isinstance(details.get("task1"), dict) else {}
        task2 = details.get("task2") if isinstance(details.get("task2"), dict) else {}
        ingredients = str(task1.get("answer") or "no ingredients selected").strip()
        caption = str(task2.get("answer") or "no caption selected").strip()
        return f"Detected {human_list(ingredients)}. Selected the caption: {caption}."
    if task == "task1":
        return f"Detected {human_list(answer)}."
    if task == "task2":
        return f"Selected the caption: {answer}."
    return answer


def top_caption_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    raw = details.get("raw") if isinstance(details.get("raw"), dict) else {}
    traces = raw.get("traces") if isinstance(raw.get("traces"), list) else []
    trace = traces[0] if traces and isinstance(traces[0], dict) else {}
    candidates = trace.get("ranked_candidates") if isinstance(trace.get("ranked_candidates"), list) else []
    rows: list[dict[str, Any]] = []
    for item in candidates[:5]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "caption": str(item.get("caption") or item.get("text") or "").strip(),
                "category": str(item.get("cat") or item.get("category") or "").strip(),
                "score": item.get("final_score", item.get("score")),
                "siglip_score": item.get("siglip_score"),
                "rerank_score": item.get("rerank_score"),
            }
        )
    return rows


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


TIMING_LABELS = {
    "load_models_and_text_sec": "Load models and text",
    "load_siglip2_sec": "Load SigLIP2",
    "load_qwen_sec": "Load VLM",
    "text_embeddings_sec": "Text embeddings",
    "siglip2_image_and_topk_sec": "Image embedding + top-k",
    "qwen_sec": "VLM generation",
    "fusion_and_selection_sec": "Fusion and selection",
    "total_image_sec": "Total image",
    "siglip_image_and_recall_sec": "SigLIP image + recall",
    "rerank_sec": "Reranker scoring",
    "final_selection_sec": "Final selection",
    "caption_embeddings_sec": "Caption embeddings",
    "load_reranker_sec": "Load reranker",
    "siglip2_candidate_filter_sec": "SigLIP2 calorie filter",
    "candidate_filter_sec": "Candidate filter",
    "calorie_filter_sec": "Calorie candidate filter",
    "browser_image_save_sec": "Read browser image",
}

HIDDEN_TIMING_KEYS = {
    "load_models_and_text_sec",
    "load_siglip2_sec",
    "load_qwen_sec",
    "load_reranker_sec",
    "text_embeddings_sec",
    "caption_embeddings_sec",
    "final_selection_sec",
}


def timing_label(key: str) -> str:
    if key in TIMING_LABELS:
        return TIMING_LABELS[key]
    text = key.removesuffix("_sec").replace("_", " ")
    return text[:1].upper() + text[1:]


def timing_rows_from_dict(timings: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    if not isinstance(timings, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key, value in timings.items():
        if str(key) in HIDDEN_TIMING_KEYS:
            continue
        number = numeric_value(value)
        if number is None:
            continue
        label = timing_label(str(key))
        if prefix:
            label = f"{prefix}: {label}"
        rows.append({"key": str(key), "label": label, "sec": number})
    return rows


def result_timing_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    raw = details.get("raw") if isinstance(details.get("raw"), dict) else {}
    task = str(result.get("task") or "")
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    rows = timing_rows_from_dict(diagnostics.get("web_timings_sec"), prefix="Input")
    rows.extend(timing_rows_from_dict(raw.get("timings_sec"), prefix="Setup" if task == "task2" else ""))
    if task == "task2":
        traces = raw.get("traces") if isinstance(raw.get("traces"), list) else []
        trace = traces[0] if traces and isinstance(traces[0], dict) else {}
        rows.extend(timing_rows_from_dict(trace.get("timings_sec")))
    return rows


def diagnostics_summary(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    power = diagnostics.get("power") if isinstance(diagnostics.get("power"), dict) else {}
    resources = diagnostics.get("resources") if isinstance(diagnostics.get("resources"), dict) else {}
    system = diagnostics.get("system") if isinstance(diagnostics.get("system"), dict) else {}
    return {
        "timings": result_timing_rows(result),
        "power": power,
        "resources": resources,
        "system": system,
    }


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    task = str(result.get("task") or "")
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    summary: dict[str, Any] = {
        "task": task,
        "mode": result.get("mode"),
        "answer": str(result.get("answer") or "").strip(),
        "latency_sec": format_latency(result.get("latency_sec")),
        "rerank_or_vlm": str(result.get("rerank_or_vlm") or "").strip(),
        "spoken_answer": spoken_answer(result),
        "backend_json": "",
        "diagnostics": diagnostics_summary(result),
        "sections": [],
    }
    if task == "both":
        task1 = details.get("task1") if isinstance(details.get("task1"), dict) else {}
        task2 = details.get("task2") if isinstance(details.get("task2"), dict) else {}
        summary["sections"] = [summarize_result(task1), summarize_result(task2)]
        return summary
    if task == "task1":
        labels = details.get("selected_labels") if isinstance(details.get("selected_labels"), list) else []
        summary["selected_labels"] = [str(item) for item in labels if str(item).strip()]
    elif task == "task2":
        summary["prediction_caption"] = details.get("prediction_caption", summary["answer"])
        summary["prediction_category"] = str(details.get("prediction_cat") or "").strip()
        summary["final_score_mode"] = str(details.get("final_score_mode") or "").strip()
        if "rerank_applied" in details:
            summary["rerank_applied"] = bool(details.get("rerank_applied"))
        if isinstance(details.get("siglip_top_gap"), (int, float)):
            summary["siglip_top_gap"] = details.get("siglip_top_gap")
        summary["top_candidates"] = top_caption_candidates(result)
    elif task == "calories":
        calories = details.get("calories") if isinstance(details.get("calories"), dict) else {}
        summary["calories"] = calories
    if isinstance(details.get("backend_json"), str):
        summary["backend_json"] = details["backend_json"]
    return summary


def mime_to_suffix(mime: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    return mapping.get(mime.lower(), ".jpg")


def save_data_url_image(image_data: str, run_dir: Path, image_name: str | None = None) -> Path:
    if not image_data or not isinstance(image_data, str):
        raise ValueError("image_data is required")
    mime = "image/jpeg"
    payload = image_data
    if image_data.startswith("data:"):
        header, sep, payload = image_data.partition(",")
        if not sep:
            raise ValueError("Invalid image data URL")
        mime = header[5:].split(";", 1)[0] or mime
    raw = base64.b64decode(payload, validate=True)
    if not raw:
        raise ValueError("Decoded image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Decoded image is too large")
    suffix = mime_to_suffix(mime)
    if image_name:
        candidate = Path(image_name).name
        file_suffix = Path(candidate).suffix.lower()
        if file_suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg" if file_suffix == ".jpeg" else file_suffix
    path = run_dir / f"browser_image{suffix}"
    path.write_bytes(raw)
    return path


def decode_data_url(value: str, max_bytes: int, default_mime: str) -> tuple[bytes, str]:
    if not value or not isinstance(value, str):
        raise ValueError("Data URL is required")
    mime = default_mime
    payload = value
    if value.startswith("data:"):
        header, sep, payload = value.partition(",")
        if not sep:
            raise ValueError("Invalid data URL")
        mime = header[5:].split(";", 1)[0] or default_mime
    raw = base64.b64decode(payload, validate=True)
    if not raw:
        raise ValueError("Decoded payload is empty")
    if len(raw) > max_bytes:
        raise ValueError("Decoded payload is too large")
    return raw, mime


def save_data_url_audio(audio_data: str, output_dir: Path) -> Path:
    raw, mime = decode_data_url(audio_data, MAX_AUDIO_BYTES, "audio/wav")
    if mime not in {"audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave"}:
        raise ValueError(f"Unsupported audio type: {mime}; expected WAV")
    path = output_dir / "voice_command.wav"
    path.write_bytes(raw)
    return path


def normalize_tts_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())[:1200]


def stage_for_message(task: str, message: str) -> str:
    text = message.lower()
    if "queue" in text or "waiting" in text:
        return "queue"
    if "browser image" in text or "captur" in text or "saved" in text or text == "image ready":
        return "input"
    if "load" in text or "model" in text or "backend ready" in text:
        return "load"
    if task == "task1":
        if "vlm" in text or "qwen" in text or "selector" in text:
            return "vlm"
        if "candidate" in text or "rank" in text or "visual" in text or "embedding" in text or "encoding image" in text:
            return "retrieve"
    elif task == "task2":
        if "caption" in text or "recall" in text:
            return "caption"
        if "rerank" in text or "score" in text:
            return "rerank"
    elif task == "calories":
        if "composition" in text or "vlm" in text:
            return "composition"
        if "calorie" in text or "portion" in text or "grams" in text:
            return "math"
    elif task == "both":
        if "task 1" in text:
            return "task1"
        if "task 2" in text:
            return "task2"
    if "format" in text or "result" in text or "output" in text:
        return "output"
    return "run"


def sample_images(image_dir: Path, image_list: Path | None, max_images: int) -> list[Path]:
    paths: list[Path] = []
    if image_list is not None and image_list.exists():
        for raw_line in image_list.read_text(encoding="utf-8").splitlines():
            name = raw_line.strip()
            if not name:
                continue
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = image_dir / candidate
            if candidate.exists() and candidate.is_file():
                paths.append(candidate)
            if len(paths) >= max_images:
                break
    if not paths and image_dir.exists():
        allowed = {".jpg", ".jpeg", ".png", ".webp"}
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in allowed:
                paths.append(path)
            if len(paths) >= max_images:
                break
    return paths


@dataclass
class JobEvent:
    stage: str
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class Job:
    id: str
    command: dict[str, Any]
    status: str = "queued"
    task: str = ""
    mode: str = "fast"
    transcript: str = ""
    run_dir: str = ""
    image_path: str = ""
    result: dict[str, Any] | None = None
    error: str = ""
    traceback_text: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: list[JobEvent] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def add_event(self, stage: str, message: str) -> None:
        with self.lock:
            self.events.append(JobEvent(stage=stage, message=message))
            self.updated_at = time.time()

    def set_status(self, status: str, *, error: str = "", traceback_text: str = "") -> None:
        with self.lock:
            self.status = status
            self.error = error
            self.traceback_text = traceback_text
            self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "task": self.task,
                "mode": self.mode,
                "transcript": self.transcript,
                "command": self.command,
                "run_dir": self.run_dir,
                "image_path": self.image_path,
                "result": self.result,
                "error": self.error,
                "traceback": self.traceback_text,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "events": [event.to_dict() for event in self.events],
            }


class DemoController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.history = ResultHistory(DEFAULT_HISTORY_ROOT)
        self.nutrition = NutritionStore(DEFAULT_NUTRITION_ROOT)
        self.jobs: dict[str, Job] = {}
        self.jobs_lock = threading.RLock()
        self.backend_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        self.stt_lock = threading.Lock()
        self.preload_lock = threading.RLock()
        self.preload_status = "idle"
        self.preload_events: list[JobEvent] = []
        self.preload_targets = list(args.preload_backends)
        self.sample_paths = sample_images(args.sample_image_dir, args.sample_images_list, args.sample_image_count)
        self.router = TaskRouter(
            TaskRouterConfig(
                hf_token=read_hf_token(),
                diagnostics_power=True,
                diagnostics_power_interval_ms=DIAGNOSTICS_POWER_INTERVAL_MS,
            )
        )
        self.speech_model_root = args.speech_model_root.expanduser()
        self.stt_model = (
            args.stt_model
            if args.stt_model is not None
            else (
                str(whisper_bundle_path(self.speech_model_root))
                if args.stt_backend == "qairt"
                else DEFAULT_WHISPER_MODEL
            )
        )
        self.tts_model = (
            args.tts_model
            if args.tts_model is not None
            else (
                piper_bundle_path(self.speech_model_root)
                if args.tts_backend == "qairt"
                else DEFAULT_PIPER_MODEL
            )
        )
        self._tts = TextToSpeech(
            backend=args.tts_backend,
            model=self.tts_model,
            config=args.tts_config,
            qnn_backend_path=args.speech_qnn_backend_path,
            qnn_shared_memory=args.speech_qnn_shared_memory,
        )
        self._stt: SpeechToText | None = None

    def start_preload(self) -> None:
        targets = self.preload_targets
        if not targets:
            with self.preload_lock:
                self.preload_status = "skipped"
            return
        thread = threading.Thread(target=self._preload_worker, args=(targets,), name="web-demo-preload", daemon=True)
        thread.start()

    def _preload_worker(self, targets: list[str]) -> None:
        with self.preload_lock:
            self.preload_status = "running"
            self.preload_events.append(JobEvent("load", f"Preloading: {', '.join(targets)}"))
        try:
            with self.backend_lock:
                self.router.preload(targets, progress=lambda message: self._add_preload_event("load", message))
            with self.preload_lock:
                self.preload_status = "ready"
                self.preload_events.append(JobEvent("load", "Warm backends ready"))
        except Exception as exc:
            with self.preload_lock:
                self.preload_status = "error"
                self.preload_events.append(JobEvent("error", f"{type(exc).__name__}: {exc}"))

    def _add_preload_event(self, stage: str, message: str) -> None:
        with self.preload_lock:
            self.preload_events.append(JobEvent(stage, message))

    def status(self) -> dict[str, Any]:
        with self.jobs_lock:
            active = sum(1 for job in self.jobs.values() if job.status in {"queued", "running"})
        with self.preload_lock:
            preload = {
                "status": self.preload_status,
                "targets": list(self.preload_targets),
                "events": [event.to_dict() for event in self.preload_events[-20:]],
            }
        tts_available, tts_message = self._tts.availability()
        stt = self._load_stt()
        return {
            "ok": True,
            "task2": {
                "final_score_mode": "siglip_guarded",
                "reranker": "Q8 Qwen3-VL-Reranker-2B via MTMD",
            },
            "diagnostics": {
                "power": True,
                "power_interval_ms": DIAGNOSTICS_POWER_INTERVAL_MS,
            },
            "preload": preload,
            "active_jobs": active,
            "sample_images": len(self.sample_paths),
            "tts": {
                "available": tts_available,
                "engine": self._tts.backend_label(),
                "model": str(self.tts_model),
                "message": tts_message,
            },
            "voice": {
                "available": stt.is_available(),
                "engine": stt.backend_label(),
                "model": str(self.stt_model),
                "message": stt.availability_message(),
            },
        }

    def tts_available(self) -> bool:
        return self._tts.is_available()

    def synthesize_tts(self, text: str) -> bytes:
        clean = normalize_tts_text(text)
        if not clean:
            raise ValueError("TTS text is empty")
        with self.tts_lock:
            return self._tts.synthesize_wav(clean)

    def _load_stt(self) -> SpeechToText:
        if self._stt is None:
            self._stt = SpeechToText(
                backend=self.args.stt_backend,
                model=self.stt_model,
                listen_seconds=WEB_VOICE_SECONDS,
                sample_rate=WEB_STT_SAMPLE_RATE,
                device="auto",
                vad_enabled=False,
                qnn_backend_path=self.args.speech_qnn_backend_path,
                qnn_shared_memory=self.args.speech_qnn_shared_memory,
                max_decode_tokens=self.args.stt_max_decode_tokens,
            )
        return self._stt

    def transcribe_voice_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.history.create_run_dir()
        audio_path = save_data_url_audio(str(payload.get("audio_data") or ""), run_dir)
        try:
            with self.stt_lock:
                transcript = self._load_stt().transcribe_wav(audio_path)
        except Exception:
            raise
        transcript = " ".join(str(transcript or "").strip().split())
        if not transcript:
            raise CommandParseError("No voice command was transcribed")
        command = parse_command(transcript)
        if command.action != "run" or command.task is None:
            raise CommandParseError("Voice command did not map to a runnable task")
        return {
            "transcript": transcript,
            "task": command.task,
            "mode": command.mode or "fast",
            "command": command.to_dict(),
            "audio_path": str(audio_path),
            "run_dir": str(run_dir),
        }

    def create_job(self, payload: dict[str, Any]) -> Job:
        command, transcript = self._command_from_payload(payload)
        image_data = str(payload.get("image_data") or "")
        image_name = str(payload.get("image_name") or "browser-image.jpg")
        if not image_data:
            raise ValueError("image_data is required")
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            task=str(command.task or ""),
            mode=str(command.mode or "fast"),
            transcript=transcript,
            command=command.to_dict(),
        )
        with self.jobs_lock:
            self.jobs[job_id] = job
            self._trim_jobs_locked()
        thread = threading.Thread(
            target=self._run_job,
            args=(job, command, image_data, image_name),
            name=f"web-demo-job-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def random_sample_path(self) -> Path | None:
        if not self.sample_paths:
            return None
        return random.choice(self.sample_paths)

    def _trim_jobs_locked(self) -> None:
        if len(self.jobs) <= MAX_JOBS:
            return
        ordered = sorted(self.jobs.values(), key=lambda job: job.created_at)
        for job in ordered[: max(0, len(ordered) - MAX_JOBS)]:
            self.jobs.pop(job.id, None)

    def _command_from_payload(self, payload: dict[str, Any]) -> tuple[DemoCommand, str]:
        transcript = str(payload.get("transcript") or "").strip()
        if transcript:
            command = parse_command(transcript)
            if command.action != "run":
                raise CommandParseError("Only run commands are supported in the web demo")
            return command, transcript
        task = str(payload.get("task") or "").strip()
        if task not in TASKS:
            raise ValueError(f"task must be one of: {', '.join(sorted(TASKS))}")
        return DemoCommand(action="run", task=task, mode="fast", text=TASK_COMMAND_TEXT[task]), ""

    def _run_job(self, job: Job, command: DemoCommand, image_data: str, image_name: str) -> None:
        job.set_status("running")
        try:
            run_dir = self.history.create_run_dir()
            job.run_dir = str(run_dir)
            job.add_event("input", "Saving browser image")
            image_save_started = time.perf_counter()
            image_path = save_data_url_image(image_data, run_dir, image_name)
            image_save_sec = time.perf_counter() - image_save_started
            job.image_path = str(image_path)
            job.add_event("input", "Image ready")

            acquired = self.backend_lock.acquire(blocking=False)
            if not acquired:
                job.add_event("queue", "Waiting for active pipeline")
                self.backend_lock.acquire()
            try:
                task = str(command.task or "")
                job.add_event(stage_for_message(task, "Starting pipeline"), "Starting pipeline")
                result = self.router.run(
                    image_path,
                    command,
                    run_dir / "backend",
                    progress=lambda message: job.add_event(stage_for_message(task, message), message),
                )
            finally:
                self.backend_lock.release()

            self._attach_web_diagnostics(result, {"browser_image_save_sec": image_save_sec})
            job.add_event("output", "Rendering output")
            presentation = summarize_result(result)
            speech = str(presentation.get("spoken_answer") or result.get("answer") or "")
            self.history.save(run_dir, command.text, command.to_dict(), image_path, result, speech)
            job.result = presentation
            job.add_event("done", "Result ready")
            job.set_status("completed")
        except Exception as exc:
            trace = traceback.format_exc()
            job.add_event("error", f"{type(exc).__name__}: {exc}")
            job.set_status("error", error=f"{type(exc).__name__}: {exc}", traceback_text=trace)

    def _attach_web_diagnostics(self, result: dict[str, Any], timings: dict[str, float]) -> None:
        diagnostics = result.setdefault("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            result["diagnostics"] = diagnostics
        web_timings = diagnostics.setdefault("web_timings_sec", {})
        if not isinstance(web_timings, dict):
            web_timings = {}
            diagnostics["web_timings_sec"] = web_timings
        for key, value in timings.items():
            web_timings[key] = float(value)

class WebHandler(BaseHTTPRequestHandler):
    server_version = "DishcoveryWebDemo/1.0"

    @property
    def controller(self) -> DemoController:
        return self.server.controller  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if not getattr(self.server, "quiet", False):  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            self._json_response(self.controller.status())
            return
        if path == "/api/nutrition/profile":
            self._json_response(self.controller.nutrition.profile_response())
            return
        if path == "/api/nutrition/ingredients":
            self._json_response(self.controller.nutrition.ingredient_options())
            return
        if path == "/api/nutrition/history":
            try:
                query = parse_qs(parsed.query)
                today = date.today()
                start = parse_date(query.get("from", [""])[0], today - timedelta(days=29)) if "from" in query else None
                end = parse_date(query.get("to", [""])[0], today) if "to" in query else None
                self._json_response(self.controller.nutrition.history(start, end))
            except ValueError as exc:
                self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/nutrition/summary":
            try:
                query = parse_qs(parsed.query)
                today = date.today()
                start = parse_date(query.get("from", [""])[0], today - timedelta(days=29))
                end = parse_date(query.get("to", [""])[0], today)
                self._json_response(self.controller.nutrition.summary(start, end))
            except ValueError as exc:
                self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/config":
            self._json_response(
                {
                    "tasks": [
                        {"id": "task1", "label": "Ingredients", "command": TASK_COMMAND_TEXT["task1"]},
                        {"id": "task2", "label": "Dish description", "command": TASK_COMMAND_TEXT["task2"]},
                        {"id": "calories", "label": "Calories", "command": TASK_COMMAND_TEXT["calories"]},
                    ],
                    "status": self.controller.status(),
                }
            )
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = self.controller.get_job(job_id)
            if job is None:
                self._json_response({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._json_response(job.to_dict())
            return
        if path == "/api/sample-image":
            self._sample_image_response(parse_qs(parsed.query))
            return
        self._static_response(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = read_json_body(self)
            if parsed.path == "/api/jobs":
                job = self.controller.create_job(payload)
                self._json_response(job.to_dict(), status=HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/nutrition/history":
                self._json_response(self.controller.nutrition.add_history_entry(payload), status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/voice-command":
                result = self.controller.transcribe_voice_command(payload)
                self._json_response(result)
                return
            if parsed.path == "/api/tts":
                audio = self.controller.synthesize_tts(str(payload.get("text") or ""))
                self._binary_response(audio, "audio/wav")
                return
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except (ValueError, CommandParseError) as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json_response({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = read_json_body(self)
            if parsed.path == "/api/nutrition/profile":
                self._json_response(self.controller.nutrition.save_profile(payload))
                return
            if parsed.path.startswith("/api/nutrition/history/"):
                entry_id = parsed.path.rsplit("/", 1)[-1]
                self._json_response(self.controller.nutrition.update_history_entry(entry_id, payload))
                return
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json_response({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/nutrition/history/"):
                entry_id = parsed.path.rsplit("/", 1)[-1]
                self._json_response(self.controller.nutrition.delete_history_entry(entry_id))
                return
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json_response({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False, default=json_default).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _binary_response(self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _static_response(self, path: str) -> None:
        if path in {"", "/"}:
            target = STATIC_DIR / "index.html"
        elif path.startswith("/static/"):
            target = STATIC_DIR / path.removeprefix("/static/")
        else:
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            resolved = target.resolve()
            if STATIC_DIR.resolve() not in resolved.parents and resolved != STATIC_DIR.resolve():
                raise FileNotFoundError
            if not resolved.exists() or not resolved.is_file():
                raise FileNotFoundError
            content = resolved.read_bytes()
        except FileNotFoundError:
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _sample_image_response(self, query: dict[str, list[str]]) -> None:
        path: Path | None = None
        index = 0
        sample_paths = self.controller.sample_paths
        if "index" in query and sample_paths:
            try:
                index = int(query.get("index", ["0"])[0])
            except ValueError:
                self._json_response({"error": "Invalid sample image index"}, status=HTTPStatus.BAD_REQUEST)
                return
            index = index % len(sample_paths)
            path = sample_paths[index]
        elif query.get("random", ["1"])[0] in {"0", "false", "no"} and sample_paths:
            path = sample_paths[0]
            index = 0
        else:
            path = self.controller.random_sample_path()
            if path is not None and sample_paths:
                try:
                    index = sample_paths.index(path)
                except ValueError:
                    index = 0
        if path is None or not path.exists():
            self._json_response({"error": "No sample image available"}, status=HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Dishcovery-Image-Index", str(index))
        self.send_header("X-Dishcovery-Image-Count", str(len(sample_paths)))
        self.send_header("X-Dishcovery-Image-Name", path.name)
        self.end_headers()
        self.wfile.write(content)


class DemoHTTPServer(ThreadingHTTPServer):
    controller: DemoController
    quiet: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser UI for the Dishcovery demo pipelines.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--preload-backends",
        nargs="*",
        choices=("task1", "task2_fast", "calories", "both"),
        default=list(DEFAULT_PRELOAD_BACKENDS),
        help="Warm these backends at startup; defaults to Task1 on the Dragonwing NPU.",
    )
    parser.add_argument(
        "--speech-model-root",
        type=Path,
        default=DEFAULT_SPEECH_MODEL_ROOT,
        help="Qualcomm AI Hub model root created by scripts/download_qairt_speech_models.py.",
    )
    parser.add_argument(
        "--stt-backend",
        choices=("qairt", "whisper"),
        default=DEFAULT_STT_BACKEND,
        help="Use Qualcomm Whisper-Base on HTP or the legacy faster-whisper backend.",
    )
    parser.add_argument(
        "--stt-model",
        help="Override the Qualcomm Whisper bundle path or faster-whisper model name.",
    )
    parser.add_argument(
        "--tts-backend",
        choices=("qairt", "piper", "disabled"),
        default="qairt",
        help="Use Qualcomm PiperTTS-EN on HTP, legacy Piper, or disable TTS.",
    )
    parser.add_argument(
        "--tts-model",
        type=Path,
        help="Override the Qualcomm Piper bundle path or legacy Piper ONNX path.",
    )
    parser.add_argument("--tts-config", type=Path, default=DEFAULT_PIPER_CONFIG)
    parser.add_argument(
        "--speech-qnn-backend-path",
        type=Path,
        help="Optional libQnnHtp.so override; defaults to the ORT-QNN 2.45 package.",
    )
    parser.add_argument("--speech-qnn-shared-memory", action="store_true")
    parser.add_argument("--stt-max-decode-tokens", type=int, default=64)
    parser.add_argument("--sample-image-dir", type=Path, default=DEFAULT_SAMPLE_IMAGE_DIR)
    parser.add_argument("--sample-images-list", type=Path, default=DEFAULT_SAMPLE_IMAGES_LIST)
    parser.add_argument("--sample-image-count", type=int, default=DEFAULT_SAMPLE_IMAGE_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = DemoController(args)
    server = DemoHTTPServer((args.host, args.port), WebHandler)
    server.controller = controller
    server.quiet = bool(args.quiet)
    controller.start_preload()
    url = f"http://{args.host}:{args.port}"
    print(f"Dishcovery web demo ready at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
