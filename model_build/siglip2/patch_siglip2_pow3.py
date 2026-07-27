#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace constant Pow(x, 3) nodes with Mul(Mul(x, x), x)."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--external-data-name", default="visual.onnx.data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(source), load_external_data=True)
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    new_nodes = []
    replaced = 0
    for index, node in enumerate(model.graph.node):
        if node.op_type == "Pow" and len(node.input) == 2:
            exponent = initializers.get(node.input[1])
            if exponent is not None and np.allclose(exponent, 3.0):
                base = node.input[0]
                output = node.output[0]
                prefix = node.name or f"pow3_{index}"
                temporary = f"{output}_pow3_square"
                new_nodes.extend(
                    [
                        helper.make_node(
                            "Mul",
                            [base, base],
                            [temporary],
                            name=f"{prefix}_square",
                        ),
                        helper.make_node(
                            "Mul",
                            [temporary, base],
                            list(node.output),
                            name=f"{prefix}_cube",
                        ),
                    ]
                )
                replaced += 1
                continue
        new_nodes.append(node)

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    remaining_pow = sum(node.op_type == "Pow" for node in model.graph.node)
    if replaced == 0:
        raise RuntimeError("No constant Pow(x, 3) nodes were found")
    if remaining_pow:
        raise RuntimeError(f"Patched graph still contains {remaining_pow} Pow nodes")

    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.external_data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    print(f"Replaced {replaced} Pow(x, 3) nodes")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
