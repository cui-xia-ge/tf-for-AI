"""Check whether a classifier uses the compact builtin INT8 OpenART path."""

import argparse
import json
from pathlib import Path

import tflite


# OpenART CNN 路线严格限制在这组 builtin 算子；新增算子必须在真实固件上重新验证。
OPENART_BUILTIN_OPS = {
    "ADD",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
    "MEAN",
    "MUL",
    "PAD",
    "SOFTMAX",
}


def _shape(tensor) -> list[int]:
    value = tensor.ShapeAsNumpy()
    return [] if isinstance(value, int) else value.tolist()


def inspect_model(path: Path) -> dict[str, object]:
    model_bytes = path.read_bytes()
    model = tflite.Model.GetRootAsModel(model_bytes, 0)
    subgraph = model.Subgraphs(0)
    opcode_names: list[str] = []
    custom_operators: list[str] = []
    for index in range(model.OperatorCodesLength()):
        opcode = model.OperatorCodes(index)
        custom = opcode.CustomCode()
        if custom:
            name = custom.decode("utf-8", errors="replace")
            custom_operators.append(name)
        else:
            name = tflite.opcode2name(opcode.BuiltinCode())
        opcode_names.append(name)
    operators = [
        opcode_names[subgraph.Operators(index).OpcodeIndex()]
        for index in range(subgraph.OperatorsLength())
    ]
    unique_operators = sorted(set(operators))

    input_tensor = subgraph.Tensors(int(subgraph.Inputs(0)))
    output_tensor = subgraph.Tensors(int(subgraph.Outputs(0)))
    float_runtime_tensors = []
    for index in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(index)
        if tensor.Type() == tflite.TensorType.FLOAT32:
            name = tensor.Name().decode("utf-8", errors="replace") if tensor.Name() else str(index)
            float_runtime_tensors.append(name)

    unexpected = sorted(set(unique_operators) - OPENART_BUILTIN_OPS)
    failures = []
    if custom_operators:
        failures.append(f"custom operators present: {custom_operators}")
    if unexpected:
        failures.append(f"operators require OpenART verification: {unexpected}")
    if input_tensor.Type() != tflite.TensorType.INT8:
        failures.append("public input is not INT8")
    if output_tensor.Type() != tflite.TensorType.INT8:
        failures.append("public output is not INT8")
    if _shape(input_tensor) != [1, 120, 120, 3]:
        failures.append(f"unexpected input shape: {_shape(input_tensor)}")
    if _shape(output_tensor) != [1, 10]:
        failures.append(f"unexpected output shape: {_shape(output_tensor)}")
    if float_runtime_tensors:
        failures.append(f"FLOAT32 tensors remain: {float_runtime_tensors}")

    return {
        "model": str(path),
        "model_size_bytes": len(model_bytes),
        "input_shape": _shape(input_tensor),
        "output_shape": _shape(output_tensor),
        "operators": unique_operators,
        "operator_count": len(operators),
        "custom_operators": custom_operators,
        "float_runtime_tensors": float_runtime_tensors,
        "passed_static_checks": not failures,
        "failures": failures,
        "hardware_verification_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = inspect_model(Path(args.model))
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    if not report["passed_static_checks"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
