#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = (
    ROOT / "external_assets/images/demo/Banana_bread_0098.jpg",
    ROOT / "external_assets/images/demo/Israeli_salad_0099.jpg",
    ROOT / "external_assets/images/demo/Fish_head_curry_0094.jpg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare split SigLIP2 ONNX stages to the full model.")
    parser.add_argument("full_model", type=Path)
    parser.add_argument("stage1", type=Path)
    parser.add_argument("stage2", type=Path)
    parser.add_argument("images", nargs="*", type=Path, default=list(DEFAULT_IMAGES))
    parser.add_argument(
        "--text-cache",
        type=Path,
        default=ROOT / "benchmark_inputs/frozen_text_embeddings/orin_siglip2_text_feats_cache.npz",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=ROOT / "reports/siglip2_fp32_reference.npz",
    )
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB").resize((384, 384), resample=Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return ((array - 0.5) / 0.5)[None, ...].astype(np.float32)


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def top5(embedding: np.ndarray, labels: np.ndarray, text_embeddings: np.ndarray) -> list[str]:
    return [str(labels[index]) for index in np.argsort(-(embedding @ text_embeddings.T))[:5]]


def main() -> None:
    args = parse_args()
    images = [path.resolve() for path in args.images]
    inputs = [preprocess(path) for path in images]

    full = make_session(args.full_model.resolve())
    full_input = full.get_inputs()[0].name
    full_output = full.get_outputs()[0].name
    references = np.concatenate(
        [full.run([full_output], {full_input: values})[0] for values in inputs], axis=0
    )
    del full

    stage1 = make_session(args.stage1.resolve())
    stage2 = make_session(args.stage2.resolve())
    stage1_input = stage1.get_inputs()[0].name
    stage1_output = stage1.get_outputs()[0].name
    stage2_input = stage2.get_inputs()[0].name
    stage2_output = stage2.get_outputs()[0].name
    split_outputs = []
    for values in inputs:
        intermediate = stage1.run([stage1_output], {stage1_input: values})[0]
        split_outputs.append(stage2.run([stage2_output], {stage2_input: intermediate})[0])
    split_embeddings = np.concatenate(split_outputs, axis=0)

    reference_norm = normalize_rows(references)
    split_norm = normalize_rows(split_embeddings)
    cache = np.load(args.text_cache, allow_pickle=False)
    labels = cache["labels"]
    text_embeddings = normalize_rows(cache["text_embeddings"])
    reports = []
    passed = True
    for index, image in enumerate(images):
        cosine = float(reference_norm[index] @ split_norm[index])
        reference_top5 = top5(reference_norm[index], labels, text_embeddings)
        split_top5 = top5(split_norm[index], labels, text_embeddings)
        overlap = len(set(reference_top5) & set(split_top5))
        passed = passed and cosine >= 0.999999 and overlap == 5
        reports.append(
            {
                "image": str(image),
                "cosine_similarity": cosine,
                "max_absolute_error": float(
                    np.max(np.abs(references[index] - split_embeddings[index]))
                ),
                "reference_top5": reference_top5,
                "split_top5": split_top5,
                "top5_overlap": overlap,
            }
        )
    result = {"passed": passed, "reports": reports}
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        embeddings=references,
        images=np.asarray([str(path) for path in images]),
    )
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
