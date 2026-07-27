from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .command_parser import normalize_command_text


DEFAULT_WHISPER_MODEL = "base.en"


class SpeechToText:
    def __init__(
        self,
        *,
        backend: str = "whisper",
        model: str | None = None,
        listen_seconds: float = 6.0,
        sample_rate: int = 16_000,
        device: str = "auto",
        vad_enabled: bool = False,
        **_: Any,
    ) -> None:
        if backend != "whisper":
            raise ValueError("Only the Whisper STT backend is supported by the web demo")
        self.backend = backend
        self.model = model or DEFAULT_WHISPER_MODEL
        self.listen_seconds = float(listen_seconds)
        self.sample_rate = int(sample_rate)
        self.device = device
        self.vad_enabled = bool(vad_enabled)
        self._whisper_model: Any | None = None
        self._whisper_device: str | None = None
        self._whisper_compute_type: str | None = None

    def is_available(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def availability_message(self) -> str:
        if self.is_available():
            return "faster-whisper ready"
        return "Install faster-whisper to enable browser voice commands"

    def backend_label(self) -> str:
        return "Whisper"

    def preload(self) -> None:
        self._preload_whisper_model()

    def _preload_whisper_model(self) -> Any:
        if self._whisper_model is not None:
            return self._whisper_model
        if not self.is_available():
            raise RuntimeError(self.availability_message())
        from faster_whisper import WhisperModel

        device = self._resolved_whisper_device()
        compute_type = "float16" if device == "cuda" else "int8"
        self._whisper_model = WhisperModel(self.model, device=device, compute_type=compute_type)
        self._whisper_device = device
        self._whisper_compute_type = compute_type
        return self._whisper_model

    def _transcribe_whisper_wav(self, wav_path: Path) -> str:
        model = self._preload_whisper_model()
        segments, _info = model.transcribe(
            str(wav_path),
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return normalize_command_text(" ".join(segment.text.strip() for segment in segments))

    def _resolved_whisper_device(self) -> str:
        if self.device != "auto":
            return self.device
        if not importlib.util.find_spec("ctranslate2"):
            return "cpu"
        try:
            import ctranslate2

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"
