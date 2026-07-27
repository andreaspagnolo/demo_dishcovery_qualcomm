#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


DEFAULT_BOUNDARY = "add_1817"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the Pow-fixed SigLIP2 visual encoder into two fixed-batch ONNX stages."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--boundary", default=DEFAULT_BOUNDARY)
    return parser.parse_args()


def remove_graph_io_from_value_info(model_path: Path) -> None:
    """Remove metadata entries that duplicate graph inputs or outputs."""
    model = onnx.load(str(model_path), load_external_data=False)
    io_names = {
        value.name for value in [*model.graph.input, *model.graph.output]
    }
    kept_value_info = [
        value for value in model.graph.value_info if value.name not in io_names
    ]
    removed = len(model.graph.value_info) - len(kept_value_info)
    if not removed:
        return

    model.graph.ClearField("value_info")
    model.graph.value_info.extend(kept_value_info)
    temporary = model_path.with_suffix(".tmp.onnx")
    onnx.save(model, str(temporary))
    os.replace(temporary, model_path)
    print(f"Removed {removed} duplicate graph IO value_info entries from {model_path}")


def freeze_batch_shape_input(model_path: Path, tensor_name: str = "val_0") -> None:
    model = onnx.load(str(model_path), load_external_data=False)
    matching_inputs = [value for value in model.graph.input if value.name == tensor_name]
    if len(matching_inputs) != 1:
        raise RuntimeError(
            f"Expected one {tensor_name!r} graph input in stage 2, found {len(matching_inputs)}"
        )
    kept_inputs = [value for value in model.graph.input if value.name != tensor_name]
    model.graph.ClearField("input")
    model.graph.input.extend(kept_inputs)
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=tensor_name)
    )

    temporary = model_path.with_suffix(".tmp.onnx")
    onnx.save(model, str(temporary))
    os.replace(temporary, model_path)


def describe(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    inputs = [value.name for value in model.graph.input]
    outputs = [value.name for value in model.graph.output]
    external_files = sorted(
        {
            item.value
            for initializer in model.graph.initializer
            for item in initializer.external_data
            if item.key == "location"
        }
    )
    total_size = path.stat().st_size
    for name in external_files:
        total_size += (path.parent / name).stat().st_size
    print(
        f"{path}: inputs={inputs}, outputs={outputs}, "
        f"nodes={len(model.graph.node)}, size={total_size / (1024 ** 3):.3f} GiB"
    )


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_dir = output_dir / "stage1"
    stage2_dir = output_dir / "stage2"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage1 = stage1_dir / "stage1.onnx"
    stage2 = stage2_dir / "stage2.onnx"
    onnx.utils.extract_model(
        source,
        stage1,
        input_names=["pixel_values"],
        output_names=[args.boundary],
        check_model=False,
        infer_shapes=False,
    )
    remove_graph_io_from_value_info(stage1)
    gc.collect()
    onnx.utils.extract_model(
        source,
        stage2,
        input_names=["val_0", args.boundary],
        output_names=["image_embeddings"],
        check_model=False,
        infer_shapes=False,
    )
    freeze_batch_shape_input(stage2)
    remove_graph_io_from_value_info(stage2)

    onnx.checker.check_model(str(stage1))
    onnx.checker.check_model(str(stage2))
    describe(stage1)
    describe(stage2)


if __name__ == "__main__":
    main()
