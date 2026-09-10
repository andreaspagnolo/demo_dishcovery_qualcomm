#!/usr/bin/env python3
"""Run an end-to-end PiperTTS -> WAV -> Whisper test on the IQ-9075 NPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import wave


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_web.backend.command_parser import normalize_command_text  # noqa: E402
from demo_web.backend.qairt_speech import (  # noqa: E402
    DEFAULT_SPEECH_MODEL_ROOT,
    QualcommPiperTTS,
    QualcommWhisper,
    piper_bundle_path,
    whisper_bundle_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_SPEECH_MODEL_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "run_outputs/speech_smoke/find-ingredients.wav",
    )
    parser.add_argument("--text", default="find ingredients")
    parser.add_argument("--max-decode-tokens", type=int, default=32)
    parser.add_argument("--qnn-backend-path", type=Path)
    parser.add_argument("--qnn-shared-memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tts = QualcommPiperTTS(
        piper_bundle_path(args.model_root),
        backend_path=args.qnn_backend_path,
        enable_shared_memory=args.qnn_shared_memory,
    )
    tts_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    audio = tts.synthesize_wav(args.text)
    tts_inference_seconds = time.perf_counter() - started
    output.write_bytes(audio)
    with wave.open(str(output), "rb") as wav_file:
        audio_seconds = wav_file.getnframes() / wav_file.getframerate()

    started = time.perf_counter()
    stt = QualcommWhisper(
        whisper_bundle_path(args.model_root),
        backend_path=args.qnn_backend_path,
        enable_shared_memory=args.qnn_shared_memory,
        max_decode_tokens=args.max_decode_tokens,
    )
    stt_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    transcript = stt.transcribe_wav(output)
    stt_inference_seconds = time.perf_counter() - started

    report = {
        "input_text": args.text,
        "transcript": transcript,
        "output_wav": str(output),
        "audio_seconds": audio_seconds,
        "tts_load_seconds": tts_load_seconds,
        "tts_inference_seconds": tts_inference_seconds,
        "stt_load_seconds": stt_load_seconds,
        "stt_inference_seconds": stt_inference_seconds,
    }
    print(json.dumps(report, indent=2))
    if normalize_command_text(transcript) != normalize_command_text(args.text):
        raise SystemExit("Round-trip transcript did not match the input text")


if __name__ == "__main__":
    main()
