"""Fine-tune a labeled MCXVision classifier without overwriting its source."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

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
    load_classifier_weights,
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
        "--pretrained",
        required=True,
        help="Input .weights.h5 file, or 'imagenet' for a MobileNetV2 teacher.",
    )
    parser.add_argument(
        "--data-dir",
        action="append",
        help="Labeled class-directory root; repeat to combine multiple roots.",
    )
    parser.add_argument("--test-dir")
    parser.add_argument("--output-dir", default="artifacts/finetune")
    parser.add_argument(
        "--architecture", choices=("teacher", "student"), default="teacher"
    )
    parser.add_argument("--mobilenet-alpha", type=float, default=1.0)
    parser.add_argument("--image-size", default="120x120")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument(
        "--unfreeze-tail-blocks",
        type=int,
        default=4,
        help="Final student conv blocks or MobileNetV2 groups; BN stays frozen.",
    )
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--representative-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed, args.deterministic)
    configure_gpu_memory_growth()

    use_imagenet = args.pretrained.lower() == "imagenet"
    pretrained = None if use_imagenet else Path(args.pretrained)
    if use_imagenet and args.architecture != "teacher":
        raise ValueError("ImageNet initialization is only valid for the teacher")
    if pretrained is not None and not pretrained.is_file():
        raise FileNotFoundError(
            f"pretrained checkpoint is required for fine-tuning: {pretrained}"
        )
    if args.unfreeze_tail_blocks < 0:
        raise ValueError("--unfreeze-tail-blocks cannot be negative")

    image_size = parse_image_size(args.image_size)
    data_dirs = default_data_dirs(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_finetuned.weights.h5"
    tflite_path = output_dir / f"box_{args.architecture}_int8.tflite"
    if pretrained is not None and pretrained.resolve() == checkpoint.resolve():
        raise ValueError("input and fine-tuned checkpoint paths must be different")

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
        backbone_weights="imagenet" if use_imagenet else None,
        mobilenet_alpha=args.mobilenet_alpha,
    )
    # Force variable creation before loading and fail loudly on incompatibility.
    model(tf.zeros((1, *image_size, 3), dtype=tf.float32), training=False)
    checkpoint_format = "imagenet" if use_imagenet else load_classifier_weights(
        model, pretrained, architecture=args.architecture
    )
    print(f"Loaded checkpoint format: {checkpoint_format}")

    warning = architecture_warning(args.architecture)
    if warning:
        print(f"WARNING: {warning}")
    baseline_validation = evaluate_keras_model(model, datasets.validation)
    baseline_f1 = float(baseline_validation["macro_f1"])
    # Seed the output with the source model. It is replaced only by a genuine
    # macro-F1 improvement, so fine-tuning can never destroy the input file.
    model.save_weights(str(checkpoint))

    histories: dict[str, dict[str, list[float]]] = {}
    set_finetune_blocks(model, trainable_tail_blocks=0)
    histories["classifier_head"], best_f1 = fit_training_phase(
        name="phase 1: classifier head",
        model=model,
        train_dataset=datasets.train,
        validation_dataset=datasets.validation,
        checkpoint=checkpoint,
        epochs=args.head_epochs,
        learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        initial_best=baseline_f1,
    )

    set_finetune_blocks(model, args.unfreeze_tail_blocks)
    histories["tail_blocks"], best_f1 = fit_training_phase(
        name=f"phase 2: final {args.unfreeze_tail_blocks} convolution blocks",
        model=model,
        train_dataset=datasets.train,
        validation_dataset=datasets.validation,
        checkpoint=checkpoint,
        epochs=args.finetune_epochs,
        learning_rate=args.finetune_learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        initial_best=best_f1,
    )

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
            "pretrained": args.pretrained,
            "checkpoint_format": checkpoint_format,
            "mobilenet_alpha": args.mobilenet_alpha,
            "data_dirs": [str(path) for path in data_dirs],
            "test_dir": args.test_dir,
            "image_size": list(image_size),
            "batch_size": args.batch_size,
            "head_epochs": args.head_epochs,
            "finetune_epochs": args.finetune_epochs,
            "unfreeze_tail_blocks": args.unfreeze_tail_blocks,
            "head_learning_rate": args.head_learning_rate,
            "finetune_learning_rate": args.finetune_learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
        },
        "warning": warning,
        "baseline_validation": baseline_validation,
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
