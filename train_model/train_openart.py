"""Train and rank compact real-capture CNNs for OpenART."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from inspect_openart import inspect_model as inspect_openart_model
from training_lib import (
    OPENART_CNN_CONFIGS,
    build_openart_classifier,
    configure_gpu_memory_growth,
    evaluate_keras_model,
    evaluate_tflite_model,
    export_int8_model,
    fit_training_phase,
    load_openart_datasets,
    set_reproducibility,
    write_json,
)


DEFAULT_DATA_DIR = Path("/home/cgcgs/718/dataset/box/real")
DEFAULT_OUTPUT_DIR = Path("artifacts/openart_real")
MAX_OPENART_MODEL_BYTES = 2_760_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-size", default="120x120")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--representative-samples", type=int, default=500)
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=tuple(OPENART_CNN_CONFIGS),
        help="Candidate variant; repeat to select a subset (default: all).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=123,
        help="Fixed real-capture train/validation split seed for every candidate.",
    )
    parser.add_argument("--refit-seed", type=int, default=456)
    parser.add_argument("--skip-refit", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def parse_image_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("image size must look like 120x120")
    width, height = (int(part) for part in parts)
    if width != 120 or height != 120:
        raise ValueError("OpenART input is fixed at 120x120")
    return height, width


def min_recall(report: dict[str, object]) -> float:
    recalls = report.get("per_class_recall")
    if not isinstance(recalls, list) or not recalls:
        return 0.0
    return min(float(value) for value in recalls)


def train_candidate(
    *,
    variant: str,
    seed: int,
    args: argparse.Namespace,
    data_dir: Path,
    output_dir: Path,
    image_size: tuple[int, int],
) -> dict[str, object]:
    # seed 控制初始化和训练随机性；split_seed 在所有候选间保持验证样本完全一致。
    set_reproducibility(seed, args.deterministic)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.weights.h5"
    tflite_path = output_dir / f"box_openart_{variant}_int8.tflite"

    datasets = load_openart_datasets(
        directory=data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
    )
    model = build_openart_classifier(
        variant=variant,
        input_shape=(*image_size, 3),
        num_classes=len(datasets.class_names),
        weight_decay=args.weight_decay,
    )
    model.summary()
    histories, best_f1 = fit_training_phase(
        name=f"OpenART {variant} seed {seed}",
        model=model,
        train_dataset=datasets.train,
        validation_dataset=datasets.validation,
        checkpoint=checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        schedule_alpha=0.01,
    )
    if not checkpoint.is_file():
        raise RuntimeError(f"candidate {variant} did not produce {checkpoint}")

    # 先评估 float，再用训练实拍 representative dataset 做完整 INT8 导出和复评。
    float_report = evaluate_keras_model(model, datasets.validation)
    export_report = export_int8_model(
        model,
        datasets.representative,
        tflite_path,
        representative_samples=args.representative_samples,
    )
    int8_report = evaluate_tflite_model(tflite_path, datasets.validation)
    openart_report = inspect_openart_model(tflite_path)
    if not openart_report["passed_static_checks"]:
        raise RuntimeError(
            f"OpenART static checks failed for {variant}: "
            f"{openart_report['failures']}"
        )
    size_bytes = int(openart_report["model_size_bytes"])
    if size_bytes > MAX_OPENART_MODEL_BYTES:
        raise RuntimeError(
            f"OpenART model {variant} is {size_bytes} bytes, above "
            f"the {MAX_OPENART_MODEL_BYTES}-byte reference"
        )

    report: dict[str, object] = {
        "config": {
            "route": "openart_real_cnn",
            "variant": variant,
            "seed": seed,
            "split_seed": args.split_seed,
            "data_dir": str(data_dir),
            "image_size": list(image_size),
            "batch_size": args.batch_size,
            "validation_fraction": args.validation_fraction,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "augmentation": {
                "brightness_limit": 0.08,
                "contrast_limit": 0.08,
                "rgb_shift": [[-3, 4], [-3, 3], [-4, 4]],
                "hsv_shift": {"hue": 2, "saturation": 4, "value": 3},
                "probabilities": {"brightness_contrast": 0.30, "color": 0.20},
                "excluded": [
                    "flip",
                    "rotation",
                    "zoom",
                    "blur",
                    "motion_blur",
                    "gaussian_noise",
                    "iso_noise",
                    "downscale",
                    "rgb565",
                    "jpeg_compression",
                ],
            },
            "train_counts": list(datasets.train_counts),
            "train_real_counts": list(datasets.train_real_counts),
            "train_shotmix_counts": list(datasets.train_shotmix_counts),
            "validation_real_counts": list(datasets.validation_real_counts),
            "representative_samples_requested": args.representative_samples,
        },
        "best_validation_macro_f1": best_f1,
        "checkpoint": str(checkpoint),
        "validation_source": "original_real_only",
        "float": float_report,
        "int8": int8_report,
        "quantization_accuracy_delta": float(
            int8_report["accuracy"] - float_report["accuracy"]
        ),
        "export": export_report,
        "openart_compatibility": openart_report,
        "history": histories,
    }
    write_json(output_dir / "report.json", report)
    return report


def ranking_key(item: dict[str, object]) -> tuple[float, float, float, int]:
    int8 = item["int8"]
    if not isinstance(int8, dict):
        raise ValueError("candidate report has no int8 metrics")
    # 选模优先级：INT8 macro-F1、最低类别召回率、accuracy，最后才比较大小。
    return (
        -float(int8["macro_f1"]),
        -min_recall(int8),
        -float(int8["accuracy"]),
        int(item["openart_compatibility"]["model_size_bytes"]),
    )


def copy_selected(
    selected_dir: Path,
    selected_report: dict[str, object],
    output_dir: Path,
    rank: list[dict[str, object]],
) -> None:
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_dir / "best.weights.h5", best_dir / "best.weights.h5")
    tflite_files = sorted(selected_dir.glob("box_openart_*_int8.tflite"))
    if len(tflite_files) != 1:
        raise RuntimeError(f"expected one candidate TFLite file in {selected_dir}")
    shutil.copy2(tflite_files[0], best_dir / "box_openart_int8.tflite")
    final_report = dict(selected_report)
    final_report["selected_candidate_dir"] = str(selected_dir)
    final_report["ranking"] = rank
    write_json(best_dir / "report.json", final_report)


def main() -> None:
    args = parse_args()
    configure_gpu_memory_growth()
    image_size = parse_image_size(args.image_size)
    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir)
    variants = args.variants or list(OPENART_CNN_CONFIGS)
    if args.epochs <= 0 or args.patience < 0:
        raise ValueError("epochs must be positive and patience cannot be negative")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"OpenART data directory does not exist: {data_dir}")

    candidate_reports: list[dict[str, object]] = []
    for variant in variants:
        candidate_reports.append(
            train_candidate(
                variant=variant,
                seed=args.seed,
                args=args,
                data_dir=data_dir,
                output_dir=output_dir / "candidates" / variant,
                image_size=image_size,
            )
        )
    candidate_reports.sort(key=ranking_key)
    rank = [
        {
            "variant": report["config"]["variant"],
            "seed": report["config"]["seed"],
            "int8_macro_f1": report["int8"]["macro_f1"],
            "int8_accuracy": report["int8"]["accuracy"],
            "min_int8_recall": min_recall(report["int8"]),
            "model_size_bytes": report["openart_compatibility"]["model_size_bytes"],
        }
        for report in candidate_reports
    ]
    selected = candidate_reports[0]
    selected_dir = Path(selected["checkpoint"]).parent

    # 对初选候选换训练 seed 复训；split_seed 不变，用于检查初始化敏感性。
    refit_report = None
    if not args.skip_refit:
        variant = str(selected["config"]["variant"])
        refit_dir = output_dir / "candidates" / f"refit_seed{args.refit_seed}_{variant}"
        refit_report = train_candidate(
            variant=variant,
            seed=args.refit_seed,
            args=args,
            data_dir=data_dir,
            output_dir=refit_dir,
            image_size=image_size,
        )
        if ranking_key(refit_report) <= ranking_key(selected):
            selected = refit_report
            selected_dir = refit_dir

    copy_selected(selected_dir, selected, output_dir, rank)
    search_report = {
        "route": "openart_real_cnn",
        "target_input": [120, 120, 3],
        "split_seed": args.split_seed,
        "candidate_reports": [report["config"]["variant"] for report in candidate_reports],
        "ranking": rank,
        "refit": refit_report,
        "selected_candidate": selected["config"]["variant"],
        "selected_seed": selected["config"]["seed"],
        "selected_report": str(output_dir / "best" / "report.json"),
    }
    write_json(output_dir / "search_report.json", search_report)
    print(json.dumps(search_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
