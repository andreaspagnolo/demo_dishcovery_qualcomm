#!/usr/bin/env python3
"""Download and prepare the IQ-9075 Whisper/Piper Qualcomm AI Hub bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_web.backend.qairt_speech import (  # noqa: E402
    DEFAULT_SPEECH_MODEL_ROOT,
    EXPECTED_QAIRT_MINOR,
    PIPER_ASSET_DIR,
    PIPER_MODEL_ID,
    QAIHM_VERSION,
    WHISPER_ASSET_DIR,
    WHISPER_MODEL_ID,
    generate_ep_context_wrappers,
)


DEVICE = "Dragonwing IQ-9075 EVK"
EXPECTED_MODEL_QAIRT = "2.45.0.260326154327"
ASSETS = (
    {
        "model_id": WHISPER_MODEL_ID,
        "runtime": "qnn_context_binary",
        "precision": "float",
        "asset_dir": WHISPER_ASSET_DIR,
    },
    {
        "model_id": PIPER_MODEL_ID,
        "runtime": "voice_ai",
        "precision": "float",
        "asset_dir": PIPER_ASSET_DIR,
    },
)
WHISPER_SUPPORT_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
WHISPER_SUPPORT_PATTERNS = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SPEECH_MODEL_ROOT,
        help="Destination model root; defaults to external_assets/models in this repository.",
    )
    parser.add_argument(
        "--qaihm-cli",
        type=Path,
        help="qai-hub-models executable; auto-detected when omitted.",
    )
    return parser.parse_args()


def resolve_cli(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(os.environ.get("QAIHM_CLI", "")) if os.environ.get("QAIHM_CLI") else None,
        REPO_ROOT / ".venv_qaihm" / "bin" / "qai-hub-models",
        Path(shutil.which("qai-hub-models") or "") if shutil.which("qai-hub-models") else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "qai-hub-models CLI not found; install qai-hub-models==0.58.0"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def bundle_is_complete(path: Path, model_id: str) -> bool:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = read_json(metadata_path)
    if metadata.get("model_id") != model_id:
        return False
    asset = next(item for item in ASSETS if item["model_id"] == model_id)
    if metadata.get("runtime") != asset["runtime"]:
        return False
    if metadata.get("precision") != asset["precision"]:
        return False
    if metadata.get("tool_versions", {}).get("qairt") != EXPECTED_MODEL_QAIRT:
        return False
    model_files = metadata.get("model_files")
    return isinstance(model_files, dict) and bool(model_files) and all(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in model_files
    )


def fetch_asset(cli: Path, output_root: Path, asset: dict[str, str], env: dict[str, str]) -> Path:
    model_parent = output_root / asset["model_id"]
    expected = model_parent / asset["asset_dir"]
    if bundle_is_complete(expected, asset["model_id"]):
        print(f"Using existing {asset['model_id']} bundle: {expected}", flush=True)
        return expected
    model_parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(cli),
        "fetch",
        asset["model_id"],
        "--runtime",
        asset["runtime"],
        "--precision",
        asset["precision"],
        "--device",
        DEVICE,
        "--sdk-version",
        f"qairt={EXPECTED_QAIRT_MINOR}",
        "--version",
        QAIHM_VERSION,
        "--extract",
        "--output-dir",
        str(model_parent),
    ]
    print(f"Downloading {asset['model_id']} to {model_parent}", flush=True)
    subprocess.run(command, check=True, env=env)
    if not bundle_is_complete(expected, asset["model_id"]):
        matches = [
            metadata.parent
            for metadata in model_parent.rglob("metadata.json")
            if read_json(metadata).get("model_id") == asset["model_id"]
        ]
        if len(matches) != 1 or not bundle_is_complete(matches[0], asset["model_id"]):
            raise RuntimeError(f"Cannot locate a complete {asset['model_id']} bundle under {model_parent}")
        expected = matches[0]
    return expected


def download_whisper_support(bundle: Path, cache_root: Path) -> Path:
    support_dir = bundle / "huggingface"
    revision_marker = support_dir / ".dishcovery_revision"
    required = (support_dir / "config.json", support_dir / "preprocessor_config.json")
    marker_matches = (
        revision_marker.is_file()
        and revision_marker.read_text(encoding="utf-8").strip() == WHISPER_SUPPORT_REVISION
    )
    if marker_matches and all(path.is_file() for path in required):
        print(f"Using existing Whisper support files: {support_dir}", flush=True)
        return support_dir
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to download Whisper support files") from exc
    print(f"Downloading Whisper tokenizer/config to {support_dir}", flush=True)
    snapshot_download(
        repo_id="openai/whisper-base",
        revision=WHISPER_SUPPORT_REVISION,
        allow_patterns=list(WHISPER_SUPPORT_PATTERNS),
        local_dir=support_dir,
        cache_dir=cache_root / "huggingface",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"Incomplete Whisper support files in {support_dir}")
    revision_marker.write_text(WHISPER_SUPPORT_REVISION + "\n", encoding="utf-8")
    return support_dir


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(output_root: Path, bundles: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for bundle in bundles:
        metadata = read_json(bundle / "metadata.json")
        artifacts = []
        for path in sorted(bundle.iterdir()):
            if path.is_file() and path.suffix in {".bin", ".onnx", ".json"}:
                artifacts.append(
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        records.append(
            {
                "model_id": metadata["model_id"],
                "runtime": metadata["runtime"],
                "precision": metadata["precision"],
                "qairt_version": metadata["tool_versions"]["qairt"],
                "path": str(bundle),
                "artifacts": artifacts,
            }
        )
    return {
        "schema_version": "dishcovery_qairt_speech_models_v1",
        "device": DEVICE,
        "qai_hub_models_version": QAIHM_VERSION,
        "whisper_support_revision": WHISPER_SUPPORT_REVISION,
        "output_root": str(output_root),
        "models": records,
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cli = resolve_cli(args.qaihm_cli)
    cache_root = output_root / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "HF_HOME": str(cache_root / "huggingface-home"),
            "HF_HUB_CACHE": str(cache_root / "huggingface-home" / "hub"),
            "TMPDIR": str(cache_root / "tmp"),
        }
    )
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    bundles = [fetch_asset(cli, output_root, asset, env) for asset in ASSETS]
    for bundle in bundles:
        wrappers = generate_ep_context_wrappers(bundle)
        print(f"Generated {len(wrappers)} EPContext wrappers in {bundle}", flush=True)
    whisper = next(path for path in bundles if read_json(path / "metadata.json")["model_id"] == WHISPER_MODEL_ID)
    download_whisper_support(whisper, cache_root)

    manifest = artifact_manifest(output_root, bundles)
    manifest_path = output_root / "speech_models.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Speech models ready: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
