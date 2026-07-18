"""Inspect and gate an eIQ Neutron-converted TFLite model."""

import argparse
import json
from pathlib import Path

import numpy as np
import tflite


TYPE_WIDTH = {
    tflite.TensorType.FLOAT32: 4,
    tflite.TensorType.INT8: 1,
    tflite.TensorType.UINT8: 1,
    tflite.TensorType.INT16: 2,
    tflite.TensorType.UINT16: 2,
    tflite.TensorType.INT32: 4,
    tflite.TensorType.UINT32: 4,
    tflite.TensorType.INT64: 8,
    tflite.TensorType.UINT64: 8,
}


def inspect_model(path: Path) -> dict[str, object]:
    model_bytes = path.read_bytes()
    model = tflite.Model.GetRootAsModel(model_bytes, 0)
    subgraph = model.Subgraphs(0)
    tensors: dict[str, dict[str, object]] = {}
    for index in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(index)
        name = tensor.Name().decode("utf-8", errors="replace") if tensor.Name() else ""
        shape_value = tensor.ShapeAsNumpy()
        shape = [] if isinstance(shape_value, int) else shape_value.tolist()
        element_count = int(np.prod(shape)) if shape else 1
        runtime_bytes = element_count * TYPE_WIDTH.get(tensor.Type(), 0)
        buffer_bytes = model.Buffers(tensor.Buffer()).DataLength()
        if name.startswith("Neutron"):
            tensors[name] = {
                "shape": shape,
                "runtime_bytes": runtime_bytes,
                "buffer_bytes": buffer_bytes,
            }

    scratch = tensors.get("NeutronScratch")
    if scratch is None:
        raise RuntimeError(
            "NeutronScratch was not found; this does not look like a converted model"
        )
    inputs = subgraph.InputsAsNumpy().tolist()
    input_runtime_bytes = 0
    for tensor_index in inputs:
        tensor = subgraph.Tensors(int(tensor_index))
        shape_value = tensor.ShapeAsNumpy()
        shape = [] if isinstance(shape_value, int) else shape_value.tolist()
        input_runtime_bytes += int(np.prod(shape)) * TYPE_WIDTH.get(tensor.Type(), 0)

    return {
        "model": str(path),
        "model_size_bytes": len(model_bytes),
        "neutron_scratch_bytes": int(scratch["runtime_bytes"]),
        "input_runtime_bytes": input_runtime_bytes,
        "scratch_plus_input_bytes": int(scratch["runtime_bytes"])
        + input_runtime_bytes,
        "neutron_tensors": tensors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--max-scratch", type=int, default=100_000)
    parser.add_argument("--max-model-bytes", type=int, default=430_000)
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_model(Path(args.model))
    failures = []
    if report["neutron_scratch_bytes"] > args.max_scratch:
        failures.append(
            f"scratch {report['neutron_scratch_bytes']} > {args.max_scratch}"
        )
    if report["model_size_bytes"] > args.max_model_bytes:
        failures.append(
            f"model {report['model_size_bytes']} > {args.max_model_bytes}"
        )
    report["limits"] = {
        "max_scratch": args.max_scratch,
        "max_model_bytes": args.max_model_bytes,
    }
    report["passed"] = not failures
    report["failures"] = failures
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
