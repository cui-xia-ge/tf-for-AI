"""Train a labeled box classifier and export a fully quantized TFLite model."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from inspect_openart import inspect_model as inspect_openart_model
from training_lib import (
    architecture_warning,
    build_classifier,
    configure_gpu_memory_growth,
    default_data_dirs,
    evaluate_keras_model,
    evaluate_tflite_model,
    export_int8_model,
    fit_training_phase,
    load_labeled_datasets,
    load_test_dataset,
    parse_image_size,
    set_finetune_blocks,
    set_reproducibility,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        action="append",
        help="Labeled class-directory root; repeat to combine multiple roots.",
    )
    parser.add_argument(
        "--test-dir",
        help="Optional untouched labeled test root. Validation is used if omitted.",
    )
    parser.add_argument("--output-dir", default="artifacts/train")
    parser.add_argument(
        "--architecture", choices=("teacher", "student"), default="teacher"
    )
    parser.add_argument(
        "--backbone-weights",
        default="auto",
        help=(
            "Teacher initialization: auto, imagenet, none, or a local "
            "MobileNetV2 no-top weights file. Student auto resolves to none."
        ),
    )
    parser.add_argument("--mobilenet-alpha", type=float, default=1.0)
    parser.add_argument("--image-size", default="120x120")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--unfreeze-tail-blocks", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.10,
        help="0.10 is a starting point; compare 0, 0.05 and 0.10 on fixed data.",
    )
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--representative-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def resolve_backbone_weights(
    architecture: str, requested: str
) -> tuple[str | None, str]:
    value = requested.strip()
    lowered = value.lower()
    if architecture == "student":
        if lowered not in ("auto", "none"):
            raise ValueError("student does not accept MobileNetV2 backbone weights")
        return None, "none"
    if lowered in ("auto", "imagenet"):
        return "imagenet", "imagenet"
    if lowered == "none":
        return None, "none"
    local_path = Path(value).expanduser()
    if not local_path.is_file():
        raise FileNotFoundError(f"backbone weights do not exist: {local_path}")
    return str(local_path), str(local_path.resolve())


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed, args.deterministic)
    configure_gpu_memory_growth()

    image_size = parse_image_size(args.image_size)
    data_dirs = default_data_dirs(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.weights.h5"
    tflite_path = output_dir / f"box_{args.architecture}_int8.tflite"
    backbone_weights, resolved_backbone_weights = resolve_backbone_weights(
        args.architecture, args.backbone_weights
    )
    if args.head_epochs < 0 or args.unfreeze_tail_blocks < 0:
        raise ValueError("head epochs and unfreeze block count cannot be negative")

    datasets = load_labeled_datasets(
        data_dirs=data_dirs,
        image_size=image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        augment=not args.no_augment,
    )
    evaluation = (
        load_test_dataset(
            Path(args.test_dir), image_size=image_size, batch_size=args.batch_size
        )
        if args.test_dir
        else datasets.validation
    )

    model = build_classifier(
        architecture=args.architecture,
        input_shape=(*image_size, 3),
        num_classes=len(datasets.class_names),
        backbone_weights=backbone_weights,
        mobilenet_alpha=args.mobilenet_alpha,
    )
    model.summary()
    warning = architecture_warning(args.architecture)
    if warning:
        print(f"WARNING: {warning}")

    histories: dict[str, dict[str, list[float]]] = {}
    if args.architecture == "teacher" and backbone_weights is not None:
        print(f"Teacher backbone initialization: {resolved_backbone_weights}")
        set_finetune_blocks(model, trainable_tail_blocks=0)
        histories["classifier_head"], best_f1 = fit_training_phase(
            name="transfer phase 1: classifier head",
            model=model,
            train_dataset=datasets.train,
            validation_dataset=datasets.validation,
            checkpoint=checkpoint,
            epochs=args.head_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
        )
        set_finetune_blocks(model, args.unfreeze_tail_blocks)
        histories["tail_blocks"], best_f1 = fit_training_phase(
            name=(
                f"transfer phase 2: final {args.unfreeze_tail_blocks} "
                "MobileNetV2 groups"
            ),
            model=model,
            train_dataset=datasets.train,
            validation_dataset=datasets.validation,
            checkpoint=checkpoint,
            epochs=args.epochs,
            learning_rate=args.finetune_learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            initial_best=best_f1,
        )
    else:
        histories["from_scratch"], best_f1 = fit_training_phase(
            name=f"from scratch: {args.architecture}",
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
        raise RuntimeError("training finished without producing a checkpoint")

    # MacroF1Checkpoint restores the selected epoch in memory before export.
    float_report = evaluate_keras_model(model, evaluation)
    export_report = export_int8_model(
        model,
        datasets.representative,
        tflite_path,
        representative_samples=args.representative_samples,
    )
    int8_report = evaluate_tflite_model(tflite_path, evaluation)
    openart_report = (
        inspect_openart_model(tflite_path)
        if args.architecture == "teacher"
        else None
    )
    if openart_report is not None and not openart_report["passed_static_checks"]:
        raise RuntimeError(
            f"teacher export failed OpenART static checks: "
            f"{openart_report['failures']}"
        )

    report = {
        "config": {
            "architecture": args.architecture,
            "backbone_weights_requested": args.backbone_weights,
            "backbone_weights_resolved": resolved_backbone_weights,
            "transfer_learning": backbone_weights is not None,
            "mobilenet_alpha": args.mobilenet_alpha,
            "data_dirs": [str(path) for path in data_dirs],
            "test_dir": args.test_dir,
            "image_size": list(image_size),
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "head_epochs": args.head_epochs,
            "unfreeze_tail_blocks": args.unfreeze_tail_blocks,
            "learning_rate": args.learning_rate,
            "finetune_learning_rate": args.finetune_learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "augmentation": not args.no_augment,
        },
        "warning": warning,
        "best_validation_macro_f1": best_f1,
        "checkpoint": str(checkpoint),
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
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
