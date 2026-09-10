"""Qualcomm AI Hub speech models running through ONNX Runtime QNN.

The public AI Hub bundles contain QNN context binaries plus a machine-readable
``metadata.json`` contract.  ``generate_ep_context_wrappers`` turns each context
into the small ONNX EPContext model expected by ONNX Runtime.  The runtime
classes below deliberately import accelerator and model-preprocessing packages
only when they are first used, so the web app can still start and report a
useful availability error on development hosts.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import importlib.util
import json
import os
from pathlib import Path
import threading
import wave
from typing import Any, Iterable

import numpy as np


QAIHM_VERSION = "0.58.0"
EXPECTED_QAIRT_MINOR = "2.45"
WHISPER_MODEL_ID = "whisper_base"
PIPER_MODEL_ID = "pipertts_en"
WHISPER_ASSET_DIR = "whisper_base-qnn_context_binary-float-qualcomm_qcs9075"
PIPER_ASSET_DIR = "pipertts_en-voice_ai-float-qualcomm_qcs9075"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEECH_MODEL_ROOT = Path(
    os.environ.get(
        "DISHCOVERY_SPEECH_MODEL_ROOT",
        str(
            _REPO_ROOT
            / "external_assets"
            / "models"
            / "qualcomm-ai-hub"
            / f"v{QAIHM_VERSION}"
        ),
    )
)
WHISPER_SAMPLE_RATE = 16_000
WHISPER_AUDIO_SECONDS = 30
WHISPER_MAX_SAMPLES = WHISPER_SAMPLE_RATE * WHISPER_AUDIO_SECONDS
WHISPER_CACHE_LENGTH = 199
WHISPER_ATTENTION_LENGTH = 200
PIPER_SAMPLE_RATE = 22_050
PIPER_MAX_INPUT_IDS = 512
PIPER_MAX_DURATION = 1_536
PIPER_DECODER_FRAMES = 40
PIPER_DECODER_OVERLAP = 12
PIPER_UPSAMPLE_FACTOR = 256

_QNN_REGISTRATION_LOCK = threading.Lock()


class QairtSpeechError(RuntimeError):
    """The installed speech stack or model bundle is invalid."""


@dataclass(frozen=True)
class BundleInfo:
    path: Path
    metadata: dict[str, Any]

    @property
    def qairt_version(self) -> str:
        return str(self.metadata.get("tool_versions", {}).get("qairt") or "")


def whisper_bundle_path(root: str | Path = DEFAULT_SPEECH_MODEL_ROOT) -> Path:
    return Path(root).expanduser() / WHISPER_MODEL_ID / WHISPER_ASSET_DIR


def piper_bundle_path(root: str | Path = DEFAULT_SPEECH_MODEL_ROOT) -> Path:
    return Path(root).expanduser() / PIPER_MODEL_ID / PIPER_ASSET_DIR


def inspect_bundle(
    path: str | Path,
    *,
    expected_model_id: str,
    expected_runtime: str,
    required_contexts: Iterable[str],
    require_wrappers: bool = True,
) -> BundleInfo:
    bundle = Path(path).expanduser().resolve()
    metadata_path = bundle / "metadata.json"
    if not metadata_path.is_file():
        raise QairtSpeechError(f"Missing Qualcomm model metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QairtSpeechError(f"Cannot read {metadata_path}: {exc}") from exc

    violations: list[str] = []
    if metadata.get("model_id") != expected_model_id:
        violations.append(
            f"model_id={metadata.get('model_id')!r}, expected {expected_model_id!r}"
        )
    if metadata.get("runtime") != expected_runtime:
        violations.append(
            f"runtime={metadata.get('runtime')!r}, expected {expected_runtime!r}"
        )
    qairt_version = str(metadata.get("tool_versions", {}).get("qairt") or "")
    if not qairt_version.startswith(f"{EXPECTED_QAIRT_MINOR}."):
        violations.append(
            f"QAIRT={qairt_version!r}, expected {EXPECTED_QAIRT_MINOR}.x"
        )

    model_files = metadata.get("model_files")
    if not isinstance(model_files, dict):
        violations.append("metadata.model_files is missing")
        model_files = {}
    for context_name in required_contexts:
        if context_name not in model_files:
            violations.append(f"metadata is missing {context_name}")
        if not (bundle / context_name).is_file():
            violations.append(f"bundle is missing {context_name}")
        wrapper_name = f"{Path(context_name).stem}.onnx"
        if require_wrappers and not (bundle / wrapper_name).is_file():
            violations.append(
                f"bundle is missing {wrapper_name}; run scripts/download_qairt_speech_models.py"
            )
    if violations:
        raise QairtSpeechError(f"Invalid Qualcomm speech bundle {bundle}: " + "; ".join(violations))
    return BundleInfo(bundle, metadata)


def generate_ep_context_wrappers(path: str | Path) -> list[Path]:
    """Generate one external EPContext ONNX wrapper per QNN ``.bin`` file."""

    bundle = Path(path).expanduser().resolve()
    metadata_path = bundle / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_files = metadata.get("model_files")
    if not isinstance(model_files, dict) or not model_files:
        raise QairtSpeechError(f"No model_files found in {metadata_path}")

    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise QairtSpeechError("Install onnx to generate QNN EPContext wrappers") from exc

    tensor_types = {
        "float16": TensorProto.FLOAT16,
        "float32": TensorProto.FLOAT,
        "int8": TensorProto.INT8,
        "uint8": TensorProto.UINT8,
        "int16": TensorProto.INT16,
        "uint16": TensorProto.UINT16,
        "int32": TensorProto.INT32,
        "uint32": TensorProto.UINT32,
        "int64": TensorProto.INT64,
        "uint64": TensorProto.UINT64,
    }

    def value_info(name: str, spec: dict[str, Any]) -> Any:
        dtype = str(spec.get("dtype"))
        shape = spec.get("shape")
        if dtype not in tensor_types or not isinstance(shape, list):
            raise QairtSpeechError(f"Unsupported tensor contract for {name}: {spec}")
        return helper.make_tensor_value_info(
            name, tensor_types[dtype], [int(value) for value in shape]
        )

    outputs: list[Path] = []
    qairt_version = str(metadata.get("tool_versions", {}).get("qairt") or "")
    for context_name, contract in model_files.items():
        if not str(context_name).endswith(".bin") or not isinstance(contract, dict):
            continue
        context_path = bundle / str(context_name)
        if not context_path.is_file() or context_path.stat().st_size == 0:
            raise FileNotFoundError(context_path)
        input_specs = contract.get("inputs")
        output_specs = contract.get("outputs")
        if not isinstance(input_specs, dict) or not isinstance(output_specs, dict):
            raise QairtSpeechError(f"Invalid input/output contract for {context_name}")
        inputs = [value_info(name, spec) for name, spec in input_specs.items()]
        graph_outputs = [value_info(name, spec) for name, spec in output_specs.items()]
        graph_name = Path(str(context_name)).stem
        node = helper.make_node(
            "EPContext",
            [item.name for item in inputs],
            [item.name for item in graph_outputs],
            name=graph_name,
            domain="com.microsoft",
            embed_mode=0,
            ep_cache_context=f"./{context_name}",
            ep_sdk_version=qairt_version,
            source="Qnn",
        )
        graph = helper.make_graph([node], graph_name, inputs, graph_outputs)
        model = helper.make_model(
            graph,
            producer_name="demo_dishcovery",
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )
        model.ir_version = 10
        onnx.checker.check_model(model)
        output_path = context_path.with_suffix(".onnx")
        onnx.save(model, output_path)
        outputs.append(output_path)
    return outputs


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _qairt_package_version() -> str:
    try:
        from onnxruntime_qnn import build_and_package_info

        return str(build_and_package_info.qnn_version)
    except Exception:
        return ""


class QnnContextSessionFactory:
    """Create strict HTP sessions for external QNN context wrappers."""

    def __init__(
        self,
        *,
        backend_path: str | Path | None = None,
        enable_shared_memory: bool = False,
    ) -> None:
        try:
            import onnxruntime as ort
            import onnxruntime_qnn as qnn_ep
        except ImportError as exc:
            raise QairtSpeechError(
                "Qualcomm speech requires onnxruntime and onnxruntime-qnn"
            ) from exc

        package_qairt = _qairt_package_version()
        if not package_qairt.startswith(f"{EXPECTED_QAIRT_MINOR}."):
            raise QairtSpeechError(
                "Qualcomm speech contexts require the QAIRT 2.45 ORT-QNN environment; "
                f"installed onnxruntime-qnn reports {package_qairt or 'unknown'}"
            )
        self.ort = ort
        self.qnn_ep = qnn_ep
        self.backend_path = (
            Path(backend_path).expanduser().resolve()
            if backend_path is not None
            else Path(qnn_ep.get_qnn_htp_path()).resolve()
        )
        if not self.backend_path.is_file():
            raise FileNotFoundError(f"QNN HTP backend not found: {self.backend_path}")
        self.enable_shared_memory = bool(enable_shared_memory)

        ep_name = qnn_ep.get_ep_name()
        with _QNN_REGISTRATION_LOCK:
            devices = [device for device in ort.get_ep_devices() if device.ep_name == ep_name]
            if not devices:
                ort.register_execution_provider_library(ep_name, qnn_ep.get_library_path())
                devices = [
                    device for device in ort.get_ep_devices() if device.ep_name == ep_name
                ]
        if not devices:
            raise QairtSpeechError("QNN execution-provider device was not discovered")
        self.devices = devices

    def create(self, wrapper: str | Path) -> Any:
        wrapper_path = Path(wrapper).expanduser().resolve()
        if not wrapper_path.is_file():
            raise FileNotFoundError(wrapper_path)
        options = self.ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_session_config_entry("ep.context_enable", "0")
        provider_options = {
            "backend_path": str(self.backend_path),
            "htp_performance_mode": "sustained_high_performance",
        }
        if self.enable_shared_memory:
            provider_options["enable_htp_shared_memory_allocator"] = "1"
        options.add_provider_for_devices(self.devices, provider_options)
        return self.ort.InferenceSession(str(wrapper_path), sess_options=options)


def _read_pcm_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        compression = wav_file.getcomptype()
    if compression != "NONE":
        raise QairtSpeechError(f"Unsupported WAV compression: {compression}")
    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise QairtSpeechError(f"Unsupported WAV sample width: {sample_width} bytes")
    if channels < 1:
        raise QairtSpeechError("WAV file has no audio channels")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)
    output_length = max(1, int(round(audio.size * target_rate / source_rate)))
    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.linspace(0, max(audio.size - 1, 0), output_length)
    return np.ascontiguousarray(
        np.interp(target_positions, source_positions, audio).astype(np.float32)
    )


class QualcommWhisper:
    """English transcription with AI Hub Whisper-Base contexts on HTP."""

    REQUIRED_CONTEXTS = ("encoder.bin", "decoder.bin")

    def __init__(
        self,
        bundle: str | Path,
        *,
        backend_path: str | Path | None = None,
        enable_shared_memory: bool = False,
        max_decode_tokens: int = 64,
    ) -> None:
        self.bundle = inspect_bundle(
            bundle,
            expected_model_id=WHISPER_MODEL_ID,
            expected_runtime="qnn_context_binary",
            required_contexts=self.REQUIRED_CONTEXTS,
        )
        if not _module_available("transformers"):
            raise QairtSpeechError(
                "Qualcomm Whisper preprocessing requires transformers"
            )
        support_dir = self.bundle.path / "huggingface"
        if not (support_dir / "config.json").is_file():
            raise QairtSpeechError(
                f"Missing local Whisper tokenizer/config files in {support_dir}; "
                "run scripts/download_qairt_speech_models.py"
            )
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            support_dir, local_files_only=True
        )
        self.tokenizer = WhisperTokenizer.from_pretrained(
            support_dir, local_files_only=True
        )
        config = json.loads((support_dir / "config.json").read_text(encoding="utf-8"))
        self.decoder_start_token_id = int(config["decoder_start_token_id"])
        self.eos_token_id = int(config["eos_token_id"])
        self.max_decode_tokens = max(4, min(int(max_decode_tokens), WHISPER_CACHE_LENGTH))
        self.forced_tokens = self._english_prompt_tokens()

        factory = QnnContextSessionFactory(
            backend_path=backend_path, enable_shared_memory=enable_shared_memory
        )
        self.encoder = factory.create(self.bundle.path / "encoder.onnx")
        self.decoder = factory.create(self.bundle.path / "decoder.onnx")
        self.cross_names = [
            name
            for layer in range(6)
            for name in (f"k_cache_cross_{layer}", f"v_cache_cross_{layer}")
        ]
        self.self_input_names = [
            name
            for layer in range(6)
            for name in (f"k_cache_self_{layer}_in", f"v_cache_self_{layer}_in")
        ]
        self.self_output_names = [
            name
            for layer in range(6)
            for name in (f"k_cache_self_{layer}_out", f"v_cache_self_{layer}_out")
        ]
        self._validate_sessions()

    @staticmethod
    def availability(bundle: str | Path) -> tuple[bool, str]:
        missing = [
            name
            for name in ("onnxruntime", "onnxruntime_qnn", "transformers")
            if not _module_available(name)
        ]
        if missing:
            return False, "Missing Python packages: " + ", ".join(missing)
        if not _qairt_package_version().startswith(f"{EXPECTED_QAIRT_MINOR}."):
            return False, "Use the QAIRT 2.45 onnxruntime-qnn environment"
        try:
            info = inspect_bundle(
                bundle,
                expected_model_id=WHISPER_MODEL_ID,
                expected_runtime="qnn_context_binary",
                required_contexts=QualcommWhisper.REQUIRED_CONTEXTS,
            )
        except (OSError, QairtSpeechError) as exc:
            return False, str(exc)
        if not (info.path / "huggingface" / "config.json").is_file():
            return False, f"Missing Whisper support files in {info.path / 'huggingface'}"
        return True, "Qualcomm Whisper-Base ready on QNN HTP"

    def _english_prompt_tokens(self) -> dict[int, int]:
        try:
            prompt = self.tokenizer.get_decoder_prompt_ids(
                language="en", task="transcribe", no_timestamps=True
            )
            return {int(position): int(token_id) for position, token_id in prompt}
        except (AttributeError, TypeError, ValueError):
            tokens = ["<|en|>", "<|transcribe|>", "<|notimestamps|>"]
            return {
                index: int(self.tokenizer.convert_tokens_to_ids(token))
                for index, token in enumerate(tokens, start=1)
            }

    def _validate_sessions(self) -> None:
        encoder_inputs = {item.name for item in self.encoder.get_inputs()}
        encoder_outputs = {item.name for item in self.encoder.get_outputs()}
        decoder_inputs = {item.name for item in self.decoder.get_inputs()}
        decoder_outputs = {item.name for item in self.decoder.get_outputs()}
        if encoder_inputs != {"input_features"} or encoder_outputs != set(self.cross_names):
            raise QairtSpeechError("Whisper encoder wrapper does not match the AI Hub contract")
        expected_decoder_inputs = {
            "input_ids",
            "position_ids",
            "attention_mask",
            *self.self_input_names,
            *self.cross_names,
        }
        if decoder_inputs != expected_decoder_inputs:
            raise QairtSpeechError("Whisper decoder inputs do not match the AI Hub contract")
        if decoder_outputs != {"logits", *self.self_output_names}:
            raise QairtSpeechError("Whisper decoder outputs do not match the AI Hub contract")

    def transcribe_wav(self, path: str | Path) -> str:
        audio, sample_rate = _read_pcm_wav(path)
        audio = _resample_audio(audio, sample_rate, WHISPER_SAMPLE_RATE)
        if not audio.size:
            return ""
        chunks = [
            audio[start : start + WHISPER_MAX_SAMPLES]
            for start in range(0, audio.size, WHISPER_MAX_SAMPLES)
        ]
        tokens: list[int] = []
        for chunk in chunks:
            tokens.extend(self._transcribe_chunk(chunk))
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def _transcribe_chunk(self, audio: np.ndarray) -> list[int]:
        features = self.feature_extractor(
            audio,
            sampling_rate=WHISPER_SAMPLE_RATE,
            return_tensors="np",
        )["input_features"]
        encoder_outputs = self.encoder.run(
            self.cross_names,
            {"input_features": np.ascontiguousarray(features, dtype=np.float16)},
        )
        cross_cache = {
            name: np.ascontiguousarray(value, dtype=np.float16)
            for name, value in zip(self.cross_names, encoder_outputs, strict=True)
        }
        self_cache: dict[str, np.ndarray] = {}
        for layer in range(6):
            self_cache[f"k_cache_self_{layer}_in"] = np.zeros(
                (8, 1, 64, WHISPER_CACHE_LENGTH), dtype=np.float16
            )
            self_cache[f"v_cache_self_{layer}_in"] = np.zeros(
                (8, 1, WHISPER_CACHE_LENGTH, 64), dtype=np.float16
            )

        output_ids = [self.decoder_start_token_id]
        attention_mask = np.full(
            (1, 1, 1, WHISPER_ATTENTION_LENGTH), -100.0, dtype=np.float16
        )
        for position in range(self.max_decode_tokens):
            attention_mask[..., WHISPER_ATTENTION_LENGTH - position - 1] = 0.0
            feed = {
                "input_ids": np.asarray([[output_ids[-1]]], dtype=np.int32),
                "position_ids": np.asarray([position], dtype=np.int32),
                "attention_mask": attention_mask,
                **self_cache,
                **cross_cache,
            }
            values = self.decoder.run(["logits", *self.self_output_names], feed)
            logits = np.asarray(values[0]).reshape(-1)
            next_position = position + 1
            next_id = self.forced_tokens.get(next_position)
            if next_id is None:
                next_id = int(np.argmax(logits))
            output_ids.append(int(next_id))
            self_cache = {
                input_name: np.ascontiguousarray(value, dtype=np.float16)
                for input_name, value in zip(
                    self.self_input_names, values[1:], strict=True
                )
            }
            if next_id == self.eos_token_id:
                break
        return output_ids


def _generate_path(duration: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """NumPy equivalent of Qualcomm's Piper duration-to-attention helper."""

    batch, _, target_length, source_length = mask.shape
    cumulative = np.cumsum(duration, axis=-1).reshape(batch * source_length)
    positions = np.arange(target_length, dtype=cumulative.dtype)
    path = (positions[None, :] < cumulative[:, None]).astype(mask.dtype)
    path = path.reshape(batch, source_length, target_length)
    previous = np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = (path - previous)[:, None].transpose(0, 1, 3, 2)
    return path * mask


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    samples = np.clip(audio, -1.0, 1.0)
    pcm = np.asarray(np.rint(samples * 32767.0), dtype="<i2")
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


class QualcommPiperTTS:
    """Piper Kusal English synthesis with AI Hub contexts on HTP."""

    REQUIRED_CONTEXTS = ("encoder.bin", "sdp.bin", "flow.bin", "decoder.bin")

    def __init__(
        self,
        bundle: str | Path,
        *,
        backend_path: str | Path | None = None,
        enable_shared_memory: bool = False,
    ) -> None:
        self.bundle = inspect_bundle(
            bundle,
            expected_model_id=PIPER_MODEL_ID,
            expected_runtime="voice_ai",
            required_contexts=self.REQUIRED_CONTEXTS,
        )
        if not _module_available("piper"):
            raise QairtSpeechError(
                "Qualcomm Piper preprocessing requires piper-tts==1.7.0"
            )
        from piper.phoneme_ids import phonemes_to_ids
        from piper.phonemize_espeak import EspeakPhonemizer

        self._phonemes_to_ids = phonemes_to_ids
        self._phonemizer = EspeakPhonemizer()
        factory = QnnContextSessionFactory(
            backend_path=backend_path, enable_shared_memory=enable_shared_memory
        )
        self.encoder = factory.create(self.bundle.path / "encoder.onnx")
        self.sdp = factory.create(self.bundle.path / "sdp.onnx")
        self.flow = factory.create(self.bundle.path / "flow.onnx")
        self.decoder = factory.create(self.bundle.path / "decoder.onnx")

    @staticmethod
    def availability(bundle: str | Path) -> tuple[bool, str]:
        missing = [
            name
            for name in ("onnxruntime", "onnxruntime_qnn", "piper")
            if not _module_available(name)
        ]
        if missing:
            return False, "Missing Python packages: " + ", ".join(missing)
        if not _qairt_package_version().startswith(f"{EXPECTED_QAIRT_MINOR}."):
            return False, "Use the QAIRT 2.45 onnxruntime-qnn environment"
        try:
            inspect_bundle(
                bundle,
                expected_model_id=PIPER_MODEL_ID,
                expected_runtime="voice_ai",
                required_contexts=QualcommPiperTTS.REQUIRED_CONTEXTS,
            )
        except (OSError, QairtSpeechError) as exc:
            return False, str(exc)
        return True, "Qualcomm PiperTTS-EN ready on QNN HTP"

    def _phoneme_id_chunks(self, text: str) -> list[list[int]]:
        sentences = self._phonemizer.phonemize("en-us", text)
        chunks: list[list[int]] = []
        # Every default Piper phoneme maps to one ID followed by one pad ID;
        # BOS, its pad, and EOS consume the remaining three positions.
        max_phonemes = (PIPER_MAX_INPUT_IDS - 3) // 2
        for sentence in sentences:
            phonemes = list(sentence)
            if phonemes and phonemes[-1] not in (".", "?", "!"):
                phonemes.append(".")
            for start in range(0, len(phonemes), max_phonemes):
                ids = list(self._phonemes_to_ids(phonemes[start : start + max_phonemes]))
                if ids:
                    chunks.append(ids[:PIPER_MAX_INPUT_IDS])
        return chunks

    def synthesize_wav(self, text: str) -> bytes:
        audio_chunks = [self._synthesize_ids(ids) for ids in self._phoneme_id_chunks(text)]
        if not audio_chunks:
            raise QairtSpeechError("Piper produced no phonemes for the supplied text")
        silence = np.zeros(int(PIPER_SAMPLE_RATE * 0.12), dtype=np.float32)
        combined: list[np.ndarray] = []
        for index, audio in enumerate(audio_chunks):
            if index:
                combined.append(silence)
            combined.append(audio)
        return _wav_bytes(np.concatenate(combined), PIPER_SAMPLE_RATE)

    def _synthesize_ids(self, phoneme_ids: list[int]) -> np.ndarray:
        actual_length = min(len(phoneme_ids), PIPER_MAX_INPUT_IDS)
        padded = np.zeros((1, PIPER_MAX_INPUT_IDS), dtype=np.int32)
        padded[0, :actual_length] = phoneme_ids[:actual_length]
        encoder_values = self.encoder.run(
            ["m_p", "logs_p", "x_encoded", "x_mask"],
            {
                "x": padded,
                "x_lengths": np.asarray([actual_length], dtype=np.int32),
            },
        )
        m_p, logs_p, x_encoded, x_mask = [
            np.ascontiguousarray(value, dtype=np.float32) for value in encoder_values
        ]
        y_lengths, w_ceil = self.sdp.run(
            ["y_lengths", "w_ceil"],
            {
                "x_encoded": x_encoded,
                "x_mask": x_mask,
                "noise_scale_w": np.asarray([0.8], dtype=np.float32),
                "length_scale": np.asarray([1.0], dtype=np.float32),
            },
        )
        output_frames = max(1, min(int(np.asarray(y_lengths).reshape(-1)[0]), PIPER_MAX_DURATION))
        y_mask = (
            np.arange(PIPER_MAX_DURATION)[None, None, :] < output_frames
        ).astype(np.float32)
        attention_mask = x_mask[:, :, None, :] * y_mask[:, :, :, None]
        attention = _generate_path(
            np.asarray(w_ceil, dtype=np.float32), attention_mask
        ).squeeze(1)
        z = self.flow.run(
            ["z"],
            {
                "attn_squeezed": np.ascontiguousarray(attention, dtype=np.float32),
                "m_p": m_p,
                "logs_p": logs_p,
                "noise_scale": np.asarray([0.667], dtype=np.float32),
                "y_mask": y_mask,
            },
        )[0]
        z = np.asarray(z, dtype=np.float32)
        decoder_length = PIPER_DECODER_FRAMES + 2 * PIPER_DECODER_OVERLAP
        z_buffer = np.zeros((1, 192, decoder_length), dtype=np.float32)
        first_length = PIPER_DECODER_FRAMES + PIPER_DECODER_OVERLAP
        z_buffer[:, :, :first_length] = z[:, :, :first_length]
        first = self.decoder.run(["audio"], {"z": z_buffer})[0]
        audio_parts = [
            np.asarray(first).reshape(-1)[: PIPER_DECODER_FRAMES * PIPER_UPSAMPLE_FACTOR]
        ]
        position = PIPER_DECODER_FRAMES
        decode_limit = min(
            output_frames,
            z.shape[2] - PIPER_DECODER_FRAMES - PIPER_DECODER_OVERLAP,
        )
        while position < decode_limit:
            z_buffer = np.ascontiguousarray(
                z[
                    :,
                    :,
                    position - PIPER_DECODER_OVERLAP : position
                    + PIPER_DECODER_FRAMES
                    + PIPER_DECODER_OVERLAP,
                ],
                dtype=np.float32,
            )
            decoded = np.asarray(
                self.decoder.run(["audio"], {"z": z_buffer})[0]
            ).reshape(-1)
            audio_parts.append(
                decoded[
                    PIPER_DECODER_OVERLAP
                    * PIPER_UPSAMPLE_FACTOR : (PIPER_DECODER_FRAMES + PIPER_DECODER_OVERLAP)
                    * PIPER_UPSAMPLE_FACTOR
                ]
            )
            position += PIPER_DECODER_FRAMES
        audio = np.concatenate(audio_parts).astype(np.float32, copy=False)
        return audio[: output_frames * PIPER_UPSAMPLE_FACTOR]
