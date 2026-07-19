"""Convert an INT8 student model for MCXN94x and enforce deployment limits.

The Neutron converter is a Windows executable.  On Linux it is run through
Wine with both model paths inside the Wine prefix.  Keeping the converter's
output on a ``C:`` drive avoids a v1.2.0 issue where a generated model cannot
be reopened reliably when its output is a ``Z:`` path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from .inspect_neutron import inspect_model
except ImportError:  # Executed as ``python train_model/convert_neutron.py``.
    from inspect_neutron import inspect_model


DEFAULT_WINE_PREFIX = Path.home() / ".cache" / "codex-neutron-wine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", required=True, help="neutron-converter executable")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--wine-prefix",
        default=str(DEFAULT_WINE_PREFIX),
        help="Wine prefix used for Linux conversion (ignored on Windows)",
    )
    parser.add_argument("--target", default="mcxn94x")
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="converter timeout in seconds"
    )
    parser.add_argument("--max-scratch", type=int, default=100_000)
    parser.add_argument("--max-model-bytes", type=int, default=430_000)
    return parser.parse_args()


def _run_converter(
    converter: Path,
    source: Path,
    output: Path,
    wine_prefix: Path,
    target: str,
    timeout: float,
) -> None:
    """Run the converter and atomically copy its C: output to ``output``."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        command = [str(converter), "--input", str(source), "--output", str(output)]
        environment = None
        temporary_root: Path | None = None
    else:
        wine_prefix = wine_prefix.expanduser().resolve()
        drive_c = wine_prefix / "drive_c"
        drive_c.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix="codex-neutron-", dir=str(drive_c))
        )
        staged_input = temporary_root / source.name
        staged_output = temporary_root / output.name
        shutil.copy2(source, staged_input)
        # A fresh directory per invocation also prevents stale converter output
        # from being mistaken for a successful run.
        windows_input = r"C:\codex-neutron-" + temporary_root.name.removeprefix(
            "codex-neutron-"
        ) + "\\" + staged_input.name
        windows_output = r"C:\codex-neutron-" + temporary_root.name.removeprefix(
            "codex-neutron-"
        ) + "\\" + staged_output.name
        command = [
            "wine",
            str(converter),
            "--input",
            windows_input,
            "--output",
            windows_output,
        ]
        environment = os.environ.copy()
        environment["WINEPREFIX"] = str(wine_prefix)

    command.extend(["--target", target, "--run-after-generate"])
    try:
        subprocess.run(command, check=True, env=environment, timeout=timeout)
        if os.name != "nt":
            assert temporary_root is not None
            staged_output = temporary_root / output.name
            if not staged_output.is_file():
                raise RuntimeError(
                    f"Neutron converter completed without creating {staged_output}"
                )
            with tempfile.NamedTemporaryFile(
                prefix=f".{output.name}.", dir=output.parent, delete=False
            ) as temporary_file:
                temporary_destination = Path(temporary_file.name)
            shutil.copyfile(staged_output, temporary_destination)
            os.replace(temporary_destination, output)
    finally:
        if os.name != "nt" and temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)


def convert_and_gate(
    *,
    converter: Path,
    source: Path,
    output: Path,
    wine_prefix: Path = DEFAULT_WINE_PREFIX,
    target: str = "mcxn94x",
    max_scratch: int = 100_000,
    max_model_bytes: int = 430_000,
    timeout: float = 900.0,
) -> dict[str, object]:
    """Convert ``source``, inspect Neutron resources, and return a gate report."""

    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    converter = Path(converter).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not converter.is_file():
        raise FileNotFoundError(converter)

    _run_converter(converter, source, output, Path(wine_prefix), target, timeout)
    report = inspect_model(output)
    failures: list[str] = []
    if report["neutron_scratch_bytes"] > max_scratch:
        failures.append(f"scratch {report['neutron_scratch_bytes']} > {max_scratch}")
    if report["model_size_bytes"] > max_model_bytes:
        failures.append(f"model {report['model_size_bytes']} > {max_model_bytes}")
    report["limits"] = {
        "max_scratch": max_scratch,
        "max_model_bytes": max_model_bytes,
        "target": target,
    }
    report["passed"] = not failures
    report["failures"] = failures
    report_path = output.with_suffix(".neutron.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    try:
        report = convert_and_gate(
            converter=Path(args.converter),
            source=Path(args.input),
            output=Path(args.output),
            wine_prefix=Path(args.wine_prefix),
            target=args.target,
            timeout=args.timeout,
            max_scratch=args.max_scratch,
            max_model_bytes=args.max_model_bytes,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        RuntimeError,
    ) as error:
        print(f"Neutron conversion failed: {error}")
        raise SystemExit(1) from error
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
