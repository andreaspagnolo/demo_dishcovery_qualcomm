from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .command_parser import normalize_command_text
from .qairt_speech import QualcommWhisper, whisper_bundle_path


DEFAULT_WHISPER_MODEL = "base.en"
DEFAULT_STT_BACKEND = "qairt"


class SpeechToText:
    def __init__(
        self,
        *,
        backend: str = DEFAULT_STT_BACKEND,
        model: str | Path | None = None,
        listen_seconds: float = 6.0,
        sample_rate: int = 16_000,
        device: str = "auto",
        vad_enabled: bool = False,
        qnn_backend_path: str | Path | None = None,
        qnn_shared_memory: bool = False,
        max_decode_tokens: int = 64,
        **_: Any,
    ) -> None:
        if backend not in {"qairt", "whisper"}:
            raise ValueError(f"Unsupported STT backend: {backend}")
        self.backend = backend
        self.model = model or (
            whisper_bundle_path() if backend == "qairt" else DEFAULT_WHISPER_MODEL
        )
        self.listen_seconds = float(listen_seconds)
        self.sample_rate = int(sample_rate)
        self.device = device
        self.vad_enabled = bool(vad_enabled)
        self.qnn_backend_path = qnn_backend_path
        self.qnn_shared_memory = bool(qnn_shared_memory)
        self.max_decode_tokens = int(max_decode_tokens)
        self._whisper_model: Any | None = None
        self._qairt_model: QualcommWhisper | None = None
        self._whisper_device: str | None = None
        self._whisper_compute_type: str | None = None

    def is_available(self) -> bool:
        if self.backend == "qairt":
            return QualcommWhisper.availability(Path(self.model))[0]
        return importlib.util.find_spec("faster_whisper") is not None

    def availability_message(self) -> str:
        if self.backend == "qairt":
            return QualcommWhisper.availability(Path(self.model))[1]
        if self.is_available():
            return "faster-whisper ready"
        return "Install faster-whisper to enable browser voice commands"

    def backend_label(self) -> str:
        return "Qualcomm Whisper-Base (QNN HTP)" if self.backend == "qairt" else "Whisper"

    def preload(self) -> None:
        if self.backend == "qairt":
            self._preload_qairt_model()
        else:
            self._preload_whisper_model()

    def _preload_qairt_model(self) -> QualcommWhisper:
        if self._qairt_model is None:
            self._qairt_model = QualcommWhisper(
                Path(self.model),
                backend_path=self.qnn_backend_path,
                enable_shared_memory=self.qnn_shared_memory,
                max_decode_tokens=self.max_decode_tokens,
            )
        return self._qairt_model

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

    def transcribe_wav(self, wav_path: Path) -> str:
        if self.backend == "qairt":
            transcript = self._preload_qairt_model().transcribe_wav(wav_path)
            return normalize_command_text(transcript)
        return self._transcribe_whisper_wav(wav_path)

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
