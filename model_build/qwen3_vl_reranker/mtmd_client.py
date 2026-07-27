from __future__ import annotations

import hashlib
import json
import select
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_INSTRUCTION = (
    "Carefully compare the food image with the candidate caption. "
    "Score high only when the visible dish, ingredients, cooking method, sauce, plating, "
    "and other visual details match the caption."
)


def _protocol_field(value: str | Path) -> str:
    """Make one value safe for the bridge's line-oriented TSV protocol."""
    return " ".join(str(value).replace("\t", " ").splitlines()).strip()


class MtmdWorkerError(RuntimeError):
    """A native-worker failure after which the current image is not valid."""

    def __init__(self, message: str, *, restarted: bool = False) -> None:
        super().__init__(message)
        self.restarted = bool(restarted)


class MtmdRequestTimeout(MtmdWorkerError):
    """The native worker did not answer before the request deadline."""


class MtmdRerankerClient:
    """Persistent client for qwen3-vl-reranker-mtmd.

    Keeping the subprocess alive avoids reloading the GGUF files and lets the
    bridge reuse vision embeddings for adjacent candidates from the same image.
    """

    def __init__(
        self,
        *,
        executable: str | Path,
        model: str | Path,
        mmproj: str | Path,
        compute: str = "npu",
        instruction: str = DEFAULT_INSTRUCTION,
        n_ctx: int = 4096,
        n_batch: int = 512,
        threads: int = 8,
        flash_attention: str = "off",
        image_min_tokens: int = -1,
        image_max_tokens: int = 512,
        max_batch_candidates: int = 4,
        force_individual_calls: bool = False,
        backend_dir: str | Path = "/home/ubuntu/.local/share/geniex/llama_cpp",
        image_size: int = 384,
        startup_timeout_sec: float = 300.0,
        request_timeout_sec: float = 60.0,
        restart_on_failure: bool = True,
        verbose: bool = False,
    ) -> None:
        self.executable = Path(executable).expanduser().resolve()
        self.model = Path(model).expanduser().resolve()
        self.mmproj = Path(mmproj).expanduser().resolve()
        self.instruction = _protocol_field(instruction)
        if max_batch_candidates <= 0:
            raise ValueError("max_batch_candidates must be greater than zero")
        self.max_batch_candidates = int(max_batch_candidates)
        if compute not in {"npu", "cpu", "hybrid"}:
            raise ValueError("compute must be one of: npu, cpu, hybrid")
        if flash_attention not in {"off", "on", "auto"}:
            raise ValueError("flash_attention must be one of: off, on, auto")
        if n_ctx <= 0 or n_batch <= 0 or threads <= 0:
            raise ValueError("n_ctx, n_batch, and threads must be greater than zero")
        if image_size != 384:
            raise ValueError("The EVK reranker image size is fixed at 384x384")
        if startup_timeout_sec <= 0 or request_timeout_sec <= 0:
            raise ValueError("startup_timeout_sec and request_timeout_sec must be positive")
        self.force_individual_calls = bool(force_individual_calls)
        for label, path in (
            ("bridge executable", self.executable),
            ("GGUF model", self.model),
            ("multimodal projector", self.mmproj),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")

        self._command = [
            str(self.executable),
            "--model",
            str(self.model),
            "--mmproj",
            str(self.mmproj),
            "--compute",
            compute,
            "--instruction",
            self.instruction,
            "--n-ctx",
            str(n_ctx),
            "--n-batch",
            str(n_batch),
            "--threads",
            str(threads),
            "--flash-attention",
            flash_attention,
            "--image-min-tokens",
            str(image_min_tokens),
            "--image-max-tokens",
            str(image_max_tokens),
            "--max-batch-candidates",
            str(self.max_batch_candidates),
            "--backend-dir",
            str(Path(backend_dir).expanduser()),
            "--interactive",
        ]
        if verbose:
            self._command.append("--verbose")

        self.compute = compute
        self.image_size = int(image_size)
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.request_timeout_sec = float(request_timeout_sec)
        self.restart_on_failure = bool(restart_on_failure)
        self.restart_count = 0
        self._verbose = bool(verbose)
        self.last_profile: dict[str, Any] = {}
        self.runtime_info: dict[str, Any] = {}
        self._process: subprocess.Popen[str] | None = None
        self._stderr_capture: Any = None
        self._prepared_image_dir: tempfile.TemporaryDirectory | None = (
            tempfile.TemporaryDirectory(prefix="dishcovery_mtmd_384_")
        )
        self._prepared_images: dict[tuple[str, int, int], Path] = {}
        self._start_process()

    def _readline_with_timeout(self, timeout_sec: float) -> str:
        process = self._process
        if process is None or process.stdout is None:
            raise MtmdWorkerError("mtmd reranker stdout is unavailable")
        deadline = time.monotonic() + timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            readable, _, _ = select.select([process.stdout.fileno()], [], [], remaining)
            if readable:
                return process.stdout.readline()

    def _start_process(self) -> None:
        self._stderr_capture = None if self._verbose else tempfile.TemporaryFile(
            mode="w+t", encoding="utf-8", errors="replace"
        )
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if self._verbose else self._stderr_capture,
            text=True,
            bufsize=1,
        )
        if self._process.stdout is None:
            self._stop_process(graceful=False)
            raise MtmdWorkerError("mtmd reranker stdout is unavailable")
        try:
            ready_line = self._readline_with_timeout(self.startup_timeout_sec)
        except TimeoutError as exc:
            diagnostics = self._stderr_tail()
            self._stop_process(graceful=False)
            detail = f"\nmtmd stderr:\n{diagnostics}" if diagnostics else ""
            raise MtmdWorkerError(
                f"mtmd reranker initialization exceeded {self.startup_timeout_sec:.1f}s{detail}"
            ) from exc
        if not ready_line:
            try:
                code = self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                code = self._process.poll()
            diagnostics = self._stderr_tail()
            self._stop_process(graceful=False)
            detail = f"\nmtmd stderr:\n{diagnostics}" if diagnostics else ""
            raise MtmdWorkerError(
                f"mtmd reranker failed during initialization (exit code {code}){detail}"
            )
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            self._stop_process(graceful=False)
            raise MtmdWorkerError(
                f"Invalid mtmd initialization response: {ready_line.rstrip()}"
            ) from exc
        if ready.get("ready") is not True:
            diagnostics = self._stderr_tail()
            self._stop_process(graceful=False)
            detail = f"\nmtmd stderr:\n{diagnostics}" if diagnostics else ""
            raise MtmdWorkerError(f"Unexpected mtmd initialization response: {ready}{detail}")
        if ready.get("batch_protocol") is not True:
            self._stop_process(graceful=False)
            raise MtmdWorkerError(
                "The mtmd bridge does not advertise batched scoring support; rebuild "
                "qwen3-vl-reranker-mtmd from the current source."
            )
        self.max_batch_candidates = min(
            self.max_batch_candidates,
            int(ready.get("max_batch_candidates", self.max_batch_candidates)),
        )
        self.runtime_info = {
            **dict(ready),
            "input_image_resolution": [self.image_size, self.image_size],
            "input_image_preprocessing": "RGB bicubic square resize in Python",
            "request_timeout_sec": self.request_timeout_sec,
            "restart_on_failure": self.restart_on_failure,
            "restart_count": self.restart_count,
        }

    @property
    def pid(self) -> int:
        if self._process is None:
            raise RuntimeError("mtmd reranker is not running")
        return self._process.pid

    def _stderr_tail(self, max_chars: int = 8000) -> str:
        capture = getattr(self, "_stderr_capture", None)
        if capture is None or capture.closed:
            return ""
        try:
            capture.flush()
            capture.seek(0, 2)
            end = capture.tell()
            capture.seek(max(0, end - max_chars))
            return capture.read().strip()
        except OSError:
            return ""

    def _exchange(self, fields: list[str]) -> dict[str, Any]:
        process = self._process
        if process is None:
            raise MtmdWorkerError("mtmd reranker is not running")
        if process.poll() is not None:
            code = process.returncode
            restarted = self._restart_worker()
            raise MtmdWorkerError(
                f"mtmd reranker exited with code {code}; restarted={restarted}",
                restarted=restarted,
            )
        if process.stdin is None or process.stdout is None:
            restarted = self._restart_worker()
            raise MtmdWorkerError(
                f"mtmd reranker pipes are unavailable; restarted={restarted}",
                restarted=restarted,
            )

        try:
            process.stdin.write("\t".join(_protocol_field(field) for field in fields) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            diagnostics = self._stderr_tail()
            restarted = self._restart_worker()
            detail = f"; mtmd stderr: {diagnostics}" if diagnostics else ""
            raise MtmdWorkerError(
                f"mtmd reranker request pipe failed; restarted={restarted}{detail}",
                restarted=restarted,
            ) from exc
        try:
            response = self._readline_with_timeout(self.request_timeout_sec)
        except TimeoutError as exc:
            diagnostics = self._stderr_tail()
            restarted = self._restart_worker()
            detail = f"\nmtmd stderr:\n{diagnostics}" if diagnostics else ""
            raise MtmdRequestTimeout(
                f"mtmd reranker request exceeded {self.request_timeout_sec:.1f}s; "
                f"restarted={restarted}{detail}",
                restarted=restarted,
            ) from exc
        if not response:
            code = process.poll()
            diagnostics = self._stderr_tail()
            restarted = self._restart_worker()
            detail = f"\nmtmd stderr:\n{diagnostics}" if diagnostics else ""
            raise MtmdWorkerError(
                f"mtmd reranker closed stdout unexpectedly (exit code {code}); "
                f"restarted={restarted}{detail}",
                restarted=restarted,
            )
        try:
            result = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from mtmd reranker: {response.rstrip()}") from exc
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result

    def _restart_worker(self) -> bool:
        if not self.restart_on_failure:
            self._stop_process(graceful=False)
            return False
        self._stop_process(graceful=False)
        try:
            self.restart_count += 1
            self._start_process()
        except Exception:
            return False
        self.runtime_info["restart_count"] = self.restart_count
        return True

    def _prepare_image_384(self, image_path: Path) -> tuple[Path, float, bool]:
        stat = image_path.stat()
        key = (str(image_path), int(stat.st_mtime_ns), int(stat.st_size))
        cached = self._prepared_images.get(key)
        if cached is not None and cached.is_file():
            return cached, 0.0, True

        if self._prepared_image_dir is None:
            raise RuntimeError("mtmd image preprocessor is closed")
        started = time.perf_counter()
        digest = hashlib.sha256("\0".join(map(str, key)).encode("utf-8")).hexdigest()[:20]
        destination = Path(self._prepared_image_dir.name) / f"{digest}_384.png"
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BICUBIC,
            )
            image.save(destination, format="PNG", compress_level=1)
        self._prepared_images[key] = destination
        return destination, (time.perf_counter() - started) * 1000.0, False

    def score_batch(
        self,
        image: str | Path,
        documents: Iterable[str],
        *,
        query: str = "",
        instruction: str | None = None,
    ) -> dict[str, Any]:
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Reranker image does not exist: {image_path}")
        texts = list(documents)
        if not texts:
            raise ValueError("documents must contain at least one candidate")
        if len(texts) > self.max_batch_candidates:
            raise ValueError(
                f"A batch has {len(texts)} candidates, but this bridge supports at most "
                f"{self.max_batch_candidates}"
            )

        prepared_image, resize_ms, resize_cache_hit = self._prepare_image_384(image_path)
        ipc_started = time.perf_counter()
        result = self._exchange(
            [
                "BATCH",
                str(prepared_image),
                query,
                instruction if instruction is not None else self.instruction,
                *texts,
            ]
        )
        ipc_round_trip_ms = (time.perf_counter() - ipc_started) * 1000.0
        native_elapsed_ms = float(result.get("elapsed_ms", 0.0))
        ipc_overhead_ms = max(0.0, ipc_round_trip_ms - native_elapsed_ms)
        request_timings = dict(result.get("timings_ms", {}))
        request_timings.update(
            {
                "image_resize_384_ms": resize_ms,
                "ipc_round_trip_ms": ipc_round_trip_ms,
                "ipc_overhead_ms": ipc_overhead_ms,
            }
        )
        result["timings_ms"] = request_timings
        result["image_preprocessing"] = {
            "source_path": str(image_path),
            "prepared_path": str(prepared_image),
            "resolution": [self.image_size, self.image_size],
            "resize": "bicubic_square",
            "color_mode": "RGB",
            "cache_hit": resize_cache_hit,
        }
        scores = result.get("results")
        if not isinstance(scores, list) or len(scores) != len(texts):
            raise RuntimeError(
                "mtmd reranker returned an invalid result count: "
                f"expected {len(texts)}, got {result}"
            )
        for score in scores:
            if not isinstance(score, dict) or "score" not in score:
                raise RuntimeError(f"mtmd reranker response has no score: {result}")
            item_timings = dict(score.get("timings_ms", request_timings))
            item_timings["ipc_round_trip_ms"] = ipc_round_trip_ms / len(scores)
            item_timings["ipc_overhead_ms"] = ipc_overhead_ms / len(scores)
            score["timings_ms"] = item_timings
        return result

    def score(
        self,
        image: str | Path,
        document: str,
        *,
        query: str = "",
        instruction: str | None = None,
    ) -> dict[str, Any]:
        batch = self.score_batch(
            image,
            [document],
            query=query,
            instruction=instruction,
        )
        return dict(batch["results"][0])

    def score_pairs(
        self,
        image_paths: Iterable[str | Path],
        candidate_texts: Iterable[str],
        _batch_size: int = 1,
    ) -> list[float]:
        profile = self.score_pairs_detailed(image_paths, candidate_texts, _batch_size)
        return [float(value) for value in profile["scores"]]

    def score_pairs_detailed(
        self,
        image_paths: Iterable[str | Path],
        candidate_texts: Iterable[str],
        _batch_size: int = 1,
    ) -> dict[str, Any]:
        """Score pairs and retain every native/IPC trace.

        ``force_individual_calls`` limits requests to one candidate while the
        bridge process, visual embedding, and image-KV prefix remain resident.
        This is the fixed-five EVK comparison protocol.
        """
        paths = list(image_paths)
        texts = list(candidate_texts)
        if len(paths) != len(texts):
            raise ValueError("image_paths and candidate_texts must have the same length")
        resolved_paths = [Path(path).expanduser().resolve() for path in paths]
        scores: list[float] = []
        pair_results: list[dict[str, Any]] = []
        requests: list[dict[str, Any]] = []

        def update_partial_profile() -> dict[str, Any]:
            profile = {
                "scores": list(scores),
                "results": list(pair_results),
                "requests": list(requests),
                "runtime": {**self.runtime_info, "restart_count": self.restart_count},
                "individual_calls": self.force_individual_calls,
                "candidate_count": len(texts),
                "completed_candidate_count": len(scores),
                "request_count": len(requests),
            }
            self.last_profile = profile
            return profile

        update_partial_profile()
        index = 0
        while index < len(texts):
            end = index + 1
            while (
                not self.force_individual_calls
                and
                end < len(texts)
                and resolved_paths[end] == resolved_paths[index]
                and end - index < self.max_batch_candidates
            ):
                end += 1
            result = self.score_batch(
                resolved_paths[index],
                texts[index:end],
            )
            request_results = [dict(item) for item in result["results"]]
            scores.extend(float(item["score"]) for item in request_results)
            pair_results.extend(request_results)
            requests.append(
                {
                    key: value
                    for key, value in result.items()
                    if key != "results"
                }
            )
            index = end
            update_partial_profile()
            print(f"Reranked image-caption pairs {end}/{len(texts)}", flush=True)
        return update_partial_profile()

    def _stop_process(self, *, graceful: bool) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            if graceful and process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=10 if graceful else 0.25)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None and not stream.closed:
                    stream.close()
        capture = getattr(self, "_stderr_capture", None)
        if capture is not None and not capture.closed:
            capture.close()
        self._process = None
        self._stderr_capture = None

    def close(self) -> None:
        self._stop_process(graceful=True)
        prepared = getattr(self, "_prepared_image_dir", None)
        if prepared is not None:
            prepared.cleanup()
            self._prepared_image_dir = None

    def __enter__(self) -> "MtmdRerankerClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
