#!/usr/bin/env python3
"""Package the compiled Task 1 W8A16 bundle for manual Drive distribution.

Uses only the standard library. No model loading, inference, or upload occurs.
The source directory is read-only; incomplete output is removed on failure.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile

BUNDLE = "Qwen3-VL-4B-Instruct-GenieX-QAIRT-W8A16"
REQUIRED = (
    "part1_of_4.bin", "part2_of_4.bin", "part3_of_4.bin", "part4_of_4.bin",
    "vision_encoder.bin", "embedding_weights.raw", "config.json",
    "genie_config.json", "text-encoder.json", "text-generator.json",
    "img-enc-htp.json", "htp_backend_ext_config.json", "metadata.json",
    "tokenizer.json", "tokenizer_config.json",
)
BLOCK_SIZE = 8 * 1024 * 1024


def package(source: Path, output_dir: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError("--source must be the complete compiled export directory")
    missing = [name for name in REQUIRED
               if not (source / name).is_file() or (source / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"Incomplete runtime bundle; missing or empty files: {', '.join(missing)}")
    output_dir = output_dir.expanduser().resolve()
    if output_dir == source or source in output_dir.parents:
        raise ValueError("--output-dir must be outside the source bundle")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{BUNDLE}.zip"
    sidecar = output_dir / f"{BUNDLE}.zip.sha256"
    if archive.exists() or sidecar.exists():
        raise FileExistsError(f"Output already exists in {output_dir}; choose another output directory")

    fd, temporary = tempfile.mkstemp(prefix=f".{BUNDLE}.", suffix=".partial", dir=output_dir)
    os.close(fd)
    temporary_path = Path(temporary)
    checksums: list[str] = []
    packaged: set[str] = set()
    try:
        # Stored ZIP avoids spending CPU recompressing already compact contexts.
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
            for path in sorted(source.rglob("*")):
                if not path.is_file() or path.relative_to(source).as_posix() == "SHA256SUMS":
                    continue
                relative = path.relative_to(source).as_posix()
                digest = hashlib.sha256()
                before = path.stat()
                print(f"Packaging {relative} ({before.st_size:,} bytes)", flush=True)
                info = zipfile.ZipInfo.from_file(path, f"{BUNDLE}/{relative}")
                with path.open("rb") as reader, output.open(info, "w", force_zip64=True) as writer:
                    while block := reader.read(BLOCK_SIZE):
                        writer.write(block)
                        digest.update(block)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Source changed during packaging: {path}")
                checksums.append(f"{digest.hexdigest()}  {relative}\n")
                packaged.add(relative)
            if missing := set(REQUIRED) - packaged:
                raise RuntimeError(f"Required files could not be enumerated: {sorted(missing)}")
            output.writestr(f"{BUNDLE}/SHA256SUMS", "".join(checksums))
        digest = hashlib.sha256()
        with temporary_path.open("rb") as reader:
            while block := reader.read(BLOCK_SIZE):
                digest.update(block)
        # Publish completed outputs only after every input has been read successfully.
        with sidecar.open("x", encoding="utf-8") as handle:
            handle.write(f"{digest.hexdigest()}  {archive.name}\n")
        try:
            temporary_path.rename(archive)
        except BaseException:
            sidecar.unlink()
            raise
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive, sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Complete compiled W8A16 / INT8-KV export (not the AIMET checkpoint)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Prefer an SSD directory with at least 6 GiB free")
    args = parser.parse_args()
    for path in package(args.source, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
