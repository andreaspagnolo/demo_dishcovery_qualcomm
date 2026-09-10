"""Text-to-speech backends used by the browser demo."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .qairt_speech import QualcommPiperTTS


class TextToSpeech:
    def __init__(
        self,
        *,
        backend: str,
        model: str | Path,
        config: str | Path | None = None,
        qnn_backend_path: str | Path | None = None,
        qnn_shared_memory: bool = False,
        **_: Any,
    ) -> None:
        if backend not in {"qairt", "piper", "disabled"}:
            raise ValueError(f"Unsupported TTS backend: {backend}")
        self.backend = backend
        self.model = Path(model).expanduser()
        self.config = Path(config).expanduser() if config is not None else None
        self.qnn_backend_path = qnn_backend_path
        self.qnn_shared_memory = bool(qnn_shared_memory)
        self._qairt: QualcommPiperTTS | None = None

    def is_available(self) -> bool:
        return self.availability()[0]

    def availability(self) -> tuple[bool, str]:
        if self.backend == "disabled":
            return False, "TTS is disabled"
        if self.backend == "qairt":
            return QualcommPiperTTS.availability(self.model)
        if not self.model.is_file():
            return False, f"Missing Piper model: {self.model}"
        if self.config is not None and not self.config.is_file():
            return False, f"Missing Piper config: {self.config}"
        if shutil.which("piper") is None:
            return False, "Install piper-tts to enable legacy Piper TTS"
        return True, "Legacy Piper ready"

    def backend_label(self) -> str:
        if self.backend == "qairt":
            return "Qualcomm PiperTTS-EN (QNN HTP)"
        if self.backend == "piper":
            return "Piper"
        return "Disabled"

    def synthesize_wav(self, text: str) -> bytes:
        available, message = self.availability()
        if not available:
            raise RuntimeError(message)
        if self.backend == "qairt":
            if self._qairt is None:
                self._qairt = QualcommPiperTTS(
                    self.model,
                    backend_path=self.qnn_backend_path,
                    enable_shared_memory=self.qnn_shared_memory,
                )
            return self._qairt.synthesize_wav(text)
        return self._synthesize_legacy_piper(text)

    def _synthesize_legacy_piper(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(
            prefix="dishcovery_web_tts_", suffix=".wav", delete=False
        ) as handle:
            wav_path = Path(handle.name)
        try:
            command = ["piper", "--model", str(self.model), "--output_file", str(wav_path)]
            if self.config is not None and self.config.is_file():
                command.extend(["--config", str(self.config)])
            subprocess.run(
                command,
                input=text + "\n",
                text=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return wav_path.read_bytes()
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or "").strip() or "no Piper output"
            raise RuntimeError(f"Piper TTS failed: {message}") from exc
        finally:
            wav_path.unlink(missing_ok=True)
