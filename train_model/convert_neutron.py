"""Convert an INT8 student model for MCXN94x and enforce deployment limits."""

import argparse
import json
import subprocess
from pathlib import Path

from inspect_neutron import inspect_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", required=True, help="neutron-converter executable")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-scratch", type=int, default=100_000)
    parser.add_argument("--max-model-bytes", type=int, default=430_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            args.converter,
            "--input",
            str(source),
            "--output",
            str(output),
            "--target",
            "mcxn94x",
            "--run-after-generate",
        ],
        check=True,
    )
    report = inspect_model(output)
    failures = []
    if report["neutron_scratch_bytes"] > args.max_scratch:
        failures.append(
            f"scratch {report['neutron_scratch_bytes']} > {args.max_scratch}"
        )
    if report["model_size_bytes"] > args.max_model_bytes:
        failures.append(
            f"model {report['model_size_bytes']} > {args.max_model_bytes}"
        )
    report["passed"] = not failures
    report["failures"] = failures
    report_path = output.with_suffix(".neutron.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
