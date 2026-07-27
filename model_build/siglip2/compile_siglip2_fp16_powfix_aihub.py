#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
from onnx import TensorProto, helper

if TYPE_CHECKING:
    import qai_hub as hub


DEFAULT_SOURCE_MODEL_ID = "mq8xk3lzn"
DEFAULT_DEVICE = "Dragonwing IQ-9075 EVK"
DEFAULT_QAIRT_VERSION = "2.45"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a Pow-fixed SigLIP2 visual encoder variant for QCS9075."
    )
    parser.add_argument("--source-model-id", default=DEFAULT_SOURCE_MODEL_ID)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--qairt-version", default=DEFAULT_QAIRT_VERSION)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("external_assets/models/siglip2_qcs9075_out/fp16_powfix_qairt245"),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        choices=(384,),
        default=384,
        help="Fixed by the Orin/EVK comparison contract.",
    )
    parser.add_argument("--vtcm-mb", type=int, default=0)
    parser.add_argument(
        "--precision-label",
        choices=("fp16", "w8a16", "w8a8"),
        default="fp16",
        help="Artifact label. Non-FP16 source models must already contain calibration encodings.",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=None,
        help="Required provenance JSON for calibrated W8A16/W8A8 source models.",
    )
    args = parser.parse_args()
    if args.vtcm_mb < 0:
        parser.error("--vtcm-mb cannot be negative")
    if args.precision_label != "fp16" and args.calibration_manifest is None:
        parser.error("calibrated variants require --calibration-manifest")
    if args.calibration_manifest is not None and not args.calibration_manifest.is_file():
        parser.error(f"calibration manifest does not exist: {args.calibration_manifest}")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(model: "hub.Model", destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(model.download(str(destination)))
    return downloaded.resolve()


def write_ep_context_wrapper(path: Path, image_size: int) -> None:
    node = helper.make_node(
        "EPContext",
        ["pixel_values"],
        ["output_0"],
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context="./model.bin",
        source="QNN",
    )
    graph = helper.make_graph(
        [node],
        "qnn-onnx-model",
        [
            helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT, [1, 3, image_size, image_size]
            )
        ],
        [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, [1, 1536])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 13
    onnx.save(model, path)


def main() -> None:
    args = parse_args()
    try:
        import qai_hub as hub
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "qai_hub is required to compile a context; install Qualcomm AI Hub's Python client."
        ) from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = hub.Device(args.device)
    source_model = hub.get_model(args.source_model_id)
    precision_option = (
        "default_graph_htp_precision=FLOAT16;"
        if args.precision_label in {"fp16", "w8a16"}
        else ""
    )
    qnn_options = (
        f"--qairt_version {args.qairt_version} "
        "--qnn_options "
        f"{precision_option}"
        "default_graph_htp_optimizations=ENABLE_DLBC_WEIGHTS=1,O=3;"
        f"default_graph_htp_vtcm_size={args.vtcm_mb}"
    )
    name = (
        f"siglip2_visual_qcs9075_{args.precision_label}_powfix_"
        f"{args.image_size}_vtcm{args.vtcm_mb}"
    )

    print(f"Source model: {source_model.model_id} ({source_model.name})", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Options: {qnn_options}", flush=True)

    compile_jobs, link_job = hub.submit_compile_and_link_jobs(
        models=source_model,
        device=device,
        input_specs={"pixel_values": ((1, 3, args.image_size, args.image_size), "float32")},
        compile_options=qnn_options,
        link_options=qnn_options,
        name=name,
    )
    if link_job is None:
        raise RuntimeError("Compile failed before the link job was created")

    for job in compile_jobs:
        print(f"Compile job: {job.job_id} {job.url}", flush=True)
    print(f"Link job: {link_job.job_id} {link_job.url}", flush=True)
    link_status = link_job.wait()
    print(link_status, flush=True)
    if "SUCCESS" not in str(link_status):
        raise RuntimeError(f"Link job failed: {link_job.url}")

    linked_model = link_job.get_target_model()
    linked_path = download_model(
        linked_model,
        args.output_dir / "model.bin",
    )

    wrapper_path = args.output_dir / "model.onnx"
    write_ep_context_wrapper(wrapper_path, args.image_size)

    profile_job = hub.submit_profile_job(
        model=linked_model,
        device=device,
        options=f"--qairt_version {args.qairt_version}",
        name=f"{name}_profile",
    )
    print(f"Profile job: {profile_job.job_id} {profile_job.url}", flush=True)
    profile_status = profile_job.wait()
    print(profile_status, flush=True)
    profile_status_text = str(profile_status)
    profile_succeeded = "SUCCESS" in profile_status_text
    profile_artifact_paths: list[Path] = []
    if profile_succeeded:
        profile_dir = args.output_dir / "profile_results"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_job.download_results(str(profile_dir))
        profile_artifact_paths = sorted(
            path for path in profile_dir.rglob("*") if path.is_file()
        )
    else:
        failure_dir = args.output_dir / "profile_failure_logs"
        failure_dir.mkdir(parents=True, exist_ok=True)
        profile_artifact_paths = [
            Path(path).resolve()
            for path in profile_job.download_job_logs(str(failure_dir))
        ]

    artifacts: list[dict[str, Any]] = [
        {"kind": "qnn_context", "path": str(linked_path), "sha256": sha256(linked_path)},
        {"kind": "qnn_context_onnx", "path": str(wrapper_path), "sha256": sha256(wrapper_path)},
    ]
    artifacts.extend(
        {
            "kind": "profile_result" if profile_succeeded else "profile_failure_log",
            "path": str(path),
            "sha256": sha256(path),
        }
        for path in profile_artifact_paths
    )
    manifest = {
        "name": name,
        "source_model_id": source_model.model_id,
        "device": args.device,
        "qairt_version": args.qairt_version,
        "precision_label": args.precision_label,
        "image_size": args.image_size,
        "vtcm_mb": args.vtcm_mb,
        "calibration_manifest": (
            str(args.calibration_manifest.resolve())
            if args.calibration_manifest is not None
            else None
        ),
        "calibration_manifest_sha256": (
            sha256(args.calibration_manifest.resolve())
            if args.calibration_manifest is not None
            else None
        ),
        "input_specs": {"pixel_values": [[1, 3, args.image_size, args.image_size], "float32"]},
        "compile_options": qnn_options,
        "compile_job_ids": [job.job_id for job in compile_jobs],
        "link_job_id": link_job.job_id,
        "context_interface": {
            "input": "pixel_values",
            "input_shape": [1, 3, args.image_size, args.image_size],
            "output": "output_0",
            "output_shape": [1, 1536],
        },
        "profile_job_id": profile_job.job_id,
        "profile_status": profile_status_text,
        "profile_succeeded": profile_succeeded,
        "artifacts": artifacts,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if not profile_succeeded:
        raise RuntimeError(
            f"Profile job failed: {profile_job.url}; "
            f"diagnostics recorded in {manifest_path}"
        )


if __name__ == "__main__":
    main()
