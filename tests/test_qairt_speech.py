from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from demo_web.backend.qairt_speech import (
    QairtSpeechError,
    _generate_path,
    _wav_bytes,
    generate_ep_context_wrappers,
    inspect_bundle,
)
from demo_web.backend.tts import TextToSpeech


class QairtSpeechBundleTests(unittest.TestCase):
    def test_generate_and_validate_ep_context_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "encoder.bin").write_bytes(b"fake-qnn-context")
            metadata = {
                "model_id": "test_speech",
                "runtime": "qnn_context_binary",
                "precision": "float",
                "tool_versions": {"qairt": "2.45.0.260326154327"},
                "model_files": {
                    "encoder.bin": {
                        "inputs": {
                            "input": {"shape": [1, 4], "dtype": "float16"}
                        },
                        "outputs": {
                            "output": {"shape": [1, 2], "dtype": "float16"}
                        },
                    }
                },
            }
            (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            wrappers = generate_ep_context_wrappers(bundle)

            self.assertEqual(wrappers, [bundle / "encoder.onnx"])
            info = inspect_bundle(
                bundle,
                expected_model_id="test_speech",
                expected_runtime="qnn_context_binary",
                required_contexts=("encoder.bin",),
            )
            self.assertEqual(info.qairt_version, "2.45.0.260326154327")

    def test_rejects_wrong_qairt_minor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "model.bin").write_bytes(b"context")
            (bundle / "model.onnx").write_bytes(b"wrapper")
            (bundle / "metadata.json").write_text(
                json.dumps(
                    {
                        "model_id": "test",
                        "runtime": "voice_ai",
                        "tool_versions": {"qairt": "2.46.0"},
                        "model_files": {"model.bin": {"inputs": {}, "outputs": {}}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(QairtSpeechError, "expected 2.45"):
                inspect_bundle(
                    bundle,
                    expected_model_id="test",
                    expected_runtime="voice_ai",
                    required_contexts=("model.bin",),
                )


class QairtPiperHelpersTests(unittest.TestCase):
    def test_generate_path_assigns_duration_frames(self) -> None:
        duration = np.asarray([[[2.0, 1.0]]], dtype=np.float32)
        mask = np.ones((1, 1, 3, 2), dtype=np.float32)
        path = _generate_path(duration, mask)
        np.testing.assert_array_equal(
            path,
            np.asarray([[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]]),
        )

    def test_wav_bytes_are_pcm16_mono(self) -> None:
        payload = _wav_bytes(np.asarray([-1.0, 0.0, 1.0], dtype=np.float32), 22_050)
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 22_050)
            self.assertEqual(wav_file.getnframes(), 3)

    def test_disabled_tts_reports_unavailable(self) -> None:
        tts = TextToSpeech(backend="disabled", model=Path("unused"))
        self.assertEqual(tts.availability(), (False, "TTS is disabled"))


if __name__ == "__main__":
    unittest.main()
