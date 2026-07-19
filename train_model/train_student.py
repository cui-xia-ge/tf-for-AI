"""Search supervised and knowledge-distilled MCX student classifiers.

The validation split is created once with ``--split-seed`` and is shared by
every candidate.  Candidates are selected by float macro-F1 for checkpointing,
then by INT8 macro-F1 for delivery.  The teacher is loaded once and remains
frozen for the complete search.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

try:
    from .convert_neutron import convert_and_gate
    from .training_lib import (
        EXPECTED_CLASSES,
        SparseCrossentropyWithLabelSmoothing,
        build_classifier,
        configure_gpu_memory_growth,
        evaluate_keras_model,
        evaluate_tflite_model,
        export_int8_model,
        load_classifier_weights,
        load_labeled_datasets,
        set_reproducibility,
        write_json,
    )
except ImportError:  # Executed as ``python train_model/train_student.py``.
    from convert_neutron import convert_and_gate
    from training_lib import (
        EXPECTED_CLASSES,
        SparseCrossentropyWithLabelSmoothing,
        build_classifier,
        configure_gpu_memory_growth,
        evaluate_keras_model,
        evaluate_tflite_model,
        export_int8_model,
        load_classifier_weights,
        load_labeled_datasets,
        set_reproducibility,
        write_json,
    )


DEFAULT_DATA_DIR = Path("/home/cgcgs/718/dataset/box/total")
DEFAULT_REPRESENTATIVE_DIR = Path("/home/cgcgs/718/dataset/box/real")
DEFAULT_TEACHER = Path("artifacts/teacher_stratified/best.weights.h5")
DEFAULT_CONVERTER = Path(
    "/media/cgcgs/7CEA04FEEA04B704/Apps/work/eIQ_Toolkit_v1.10.0/"
    "bin/neutron-converter/v1.2.0/neutron-converter.exe"
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    temperature: float
    hard_loss_weight: float
    distillation: bool


def candidate_specs() -> list[CandidateSpec]:
    specs = [CandidateSpec("supervised", 1.0, 1.0, False)]
    for temperature in (2.0, 4.0, 6.0):
        for hard_loss_weight in (0.5, 0.7):
            specs.append(
                CandidateSpec(
                    name=f"distill_t{int(temperature)}_hard{hard_loss_weight:g}",
                    temperature=temperature,
                    hard_loss_weight=hard_loss_weight,
                    distillation=True,
                )
            )
    return specs


class DistillationModel(tf.keras.Model):
    """Keras wrapper whose public call is always the student logits."""

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        temperature: float,
        hard_loss_weight: float,
        distillation: bool,
        label_smoothing: float,
    ) -> None:
        super().__init__(name=f"distiller_{student.name}")
        self.student = student
        self.teacher = teacher
        self.temperature = float(temperature)
        self.hard_loss_weight = float(hard_loss_weight)
        self.distillation = bool(distillation)
        self.hard_loss = SparseCrossentropyWithLabelSmoothing(
            label_smoothing=label_smoothing, from_logits=True
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.hard_loss_tracker = tf.keras.metrics.Mean(name="hard_loss")
        self.distill_loss_tracker = tf.keras.metrics.Mean(name="distill_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.hard_loss_tracker,
            self.distill_loss_tracker,
            self.accuracy_tracker,
        ]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def _compute_losses(self, images, labels, training: bool):
        student_logits = self.student(images, training=training)
        hard = tf.reduce_mean(self.hard_loss(tf.cast(labels, tf.int32), student_logits))
        if self.distillation:
            teacher_logits = tf.stop_gradient(self.teacher(images, training=False))
            temperature = tf.cast(self.temperature, student_logits.dtype)
            teacher_probabilities = tf.nn.softmax(teacher_logits / temperature)
            student_log_probabilities = tf.nn.log_softmax(student_logits / temperature)
            soft = -tf.reduce_mean(
                tf.reduce_sum(
                    teacher_probabilities * student_log_probabilities, axis=-1
                )
            ) * tf.square(temperature)
        else:
            soft = tf.zeros((), dtype=hard.dtype)
        total = self.hard_loss_weight * hard
        if self.distillation:
            total += (1.0 - self.hard_loss_weight) * soft
        if self.student.losses:
            total += tf.add_n(self.student.losses)
        return total, hard, soft, student_logits

    def train_step(self, data):
        images, labels = data[0], data[1]
        with tf.GradientTape() as tape:
            total, hard, soft, logits = self._compute_losses(images, labels, training=True)
        gradients = tape.gradient(total, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self.loss_tracker.update_state(total)
        self.hard_loss_tracker.update_state(hard)
        self.distill_loss_tracker.update_state(soft)
        self.accuracy_tracker.update_state(labels, logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        images, labels = data[0], data[1]
        total, hard, soft, logits = self._compute_losses(images, labels, training=False)
        self.loss_tracker.update_state(total)
        self.hard_loss_tracker.update_state(hard)
        self.distill_loss_tracker.update_state(soft)
        self.accuracy_tracker.update_state(labels, logits)
        return {metric.name: metric.result() for metric in self.metrics}


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Checkpoint only student weights according to validation macro-F1."""

    def __init__(
        self,
        validation: tf.data.Dataset,
        filepath: Path,
        initial_best: float = -1.0,
        min_delta: float = 1e-5,
    ) -> None:
        super().__init__()
        self.validation = validation
        self.filepath = Path(filepath)
        self.best_f1 = float(initial_best)
        self.min_delta = float(min_delta)
        self.best_weights: list[np.ndarray] | None = None

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.student.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        report = evaluate_keras_model(self.model, self.validation)
        current_f1 = float(report["macro_f1"])
        if logs is not None:
            logs["val_macro_f1"] = current_f1
        if current_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = current_f1
            self.best_weights = self.model.student.get_weights()
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.model.student.save_weights(str(self.filepath))
            print(f"val_macro_f1 improved to {current_f1:.6f}; checkpoint saved")

    def on_train_end(self, logs=None):
        if self.best_weights is None:
            return
        self.model.student.set_weights(self.best_weights)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.student.save_weights(str(self.filepath))


def _parse_image_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("image size must look like 120x120")
    height, width = (int(part) for part in parts)
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return height, width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    parser.add_argument("--data-dir", action="append")
    parser.add_argument("--representative-dir", default=str(DEFAULT_REPRESENTATIVE_DIR))
    parser.add_argument("--converter", default=str(DEFAULT_CONVERTER))
    parser.add_argument("--wine-prefix", default=str(Path.home() / ".cache" / "codex-neutron-wine"))
    parser.add_argument("--output-dir", default="artifacts/student_best")
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--train-seed", type=int, default=123)
    parser.add_argument("--refit-seed", type=int, default=456)
    parser.add_argument("--image-size", default="120x120")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--representative-samples", type=int, default=500)
    parser.add_argument("--max-scratch", type=int, default=100_000)
    parser.add_argument("--max-model-bytes", type=int, default=430_000)
    parser.add_argument("--converter-timeout", type=float, default=900.0)
    parser.add_argument(
        "--initial-candidates",
        type=int,
        default=7,
        help="Useful for smoke tests; production uses all seven initial candidates.",
    )
    parser.add_argument("--skip-neutron", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _make_optimizer(
    train_dataset: tf.data.Dataset, epochs: int, learning_rate: float, weight_decay: float
):
    steps = int(tf.data.experimental.cardinality(train_dataset).numpy())
    if steps <= 0:
        raise RuntimeError("training dataset cardinality must be finite and non-empty")
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        learning_rate, max(1, steps * epochs), alpha=0.01
    )
    optimizer = tf.keras.optimizers.AdamW(schedule, weight_decay=weight_decay)
    if hasattr(optimizer, "exclude_from_weight_decay"):
        optimizer.exclude_from_weight_decay(var_names=["bias", "beta", "gamma"])
    return optimizer


def _train_candidate(
    *,
    spec: CandidateSpec,
    seed: int,
    teacher: tf.keras.Model,
    datasets,
    output_dir: Path,
    image_size: tuple[int, int],
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    representative_samples: int,
) -> dict[str, object]:
    set_reproducibility(seed)
    candidate_dir = output_dir / "candidates" / spec.name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate_dir / "best.weights.h5"
    tflite_path = candidate_dir / "box_student_int8.tflite"
    student = build_classifier(
        architecture="student",
        input_shape=(*image_size, 3),
        num_classes=len(EXPECTED_CLASSES),
    )
    distiller = DistillationModel(
        student=student,
        teacher=teacher,
        temperature=spec.temperature,
        hard_loss_weight=spec.hard_loss_weight,
        distillation=spec.distillation,
        label_smoothing=label_smoothing,
    )
    distiller.compile(
        optimizer=_make_optimizer(datasets.train, epochs, learning_rate, weight_decay)
    )
    macro_f1 = MacroF1Checkpoint(datasets.validation, checkpoint)
    callbacks = [
        macro_f1,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=min(patience, max(1, epochs))
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    print(
        f"\n===== {spec.name} (seed={seed}, T={spec.temperature:g}, "
        f"hard={spec.hard_loss_weight:g}) ====="
    )
    history = distiller.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    if not checkpoint.is_file():
        raise RuntimeError(f"candidate {spec.name} did not produce {checkpoint}")
    student.load_weights(str(checkpoint))
    float_report = evaluate_keras_model(student, datasets.validation)
    float_real_capture_report = evaluate_keras_model(student, datasets.representative)
    export_report = export_int8_model(
        student,
        datasets.representative,
        tflite_path,
        representative_samples=representative_samples,
    )
    int8_report = evaluate_tflite_model(tflite_path, datasets.validation)
    int8_real_capture_report = evaluate_tflite_model(
        tflite_path, datasets.representative
    )
    report = {
        "candidate": {
            "name": spec.name,
            "temperature": spec.temperature,
            "hard_loss_weight": spec.hard_loss_weight,
            "distillation": spec.distillation,
            "seed": seed,
        },
        "checkpoint": str(checkpoint),
        "float": float_report,
        "int8": int8_report,
        "float_real_capture": float_real_capture_report,
        "int8_real_capture": int8_real_capture_report,
        "quantization_accuracy_delta": float(
            int8_report["accuracy"] - float_report["accuracy"]
        ),
        "real_capture_quantization_accuracy_delta": float(
            int8_real_capture_report["accuracy"]
            - float_real_capture_report["accuracy"]
        ),
        "export": export_report,
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "train_counts": list(datasets.train_counts),
        "validation_counts": list(datasets.validation_counts),
    }
    write_json(candidate_dir / "report.json", report)
    del distiller, student
    gc.collect()
    return report


def _rank_key(report: dict[str, object]):
    int8 = report["int8"]
    recalls = int8["per_class_recall"]
    export = report.get("neutron") or {}
    model_size = export.get(
        "model_size_bytes", (report.get("export") or {}).get("size_bytes", 10**12)
    )
    return (
        float(int8["macro_f1"]),
        float(int8["accuracy"]),
        float(min(recalls)),
        -int(model_size),
    )


def _float_rank_key(report: dict[str, object]):
    float_report = report["float"]
    return (
        float(float_report["macro_f1"]),
        float(float_report["accuracy"]),
        float(min(float_report["per_class_recall"])),
    )


def _convert_ranked_candidates(
    reports: list[dict[str, object]],
    *,
    converter: Path,
    wine_prefix: Path,
    max_scratch: int,
    max_model_bytes: int,
    converter_timeout: float,
    skip_neutron: bool,
) -> list[dict[str, object]]:
    ranked = sorted(reports, key=_rank_key, reverse=True)
    if skip_neutron:
        for report in ranked:
            report["neutron"] = {"passed": None, "skipped": True}
        return ranked
    for report in ranked:
        candidate_dir = Path(report["checkpoint"]).parent
        source = candidate_dir / "box_student_int8.tflite"
        output = candidate_dir / "box_student_npu.tflite"
        try:
            neutron = convert_and_gate(
                converter=converter,
                source=source,
                output=output,
                wine_prefix=wine_prefix,
                timeout=converter_timeout,
                max_scratch=max_scratch,
                max_model_bytes=max_model_bytes,
            )
            report["neutron"] = neutron
            write_json(candidate_dir / "report.json", report)
            if neutron["passed"]:
                report["selected_by_gate"] = True
                write_json(candidate_dir / "report.json", report)
                break
        except Exception as error:  # Continue to the next INT8-ranked candidate.
            report["neutron"] = {"passed": False, "error": repr(error)}
            write_json(candidate_dir / "report.json", report)
    return ranked


def main() -> None:
    args = parse_args()
    if args.initial_candidates < 1 or args.initial_candidates > 7:
        raise ValueError("initial-candidates must be between 1 and 7")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    set_reproducibility(args.train_seed, args.deterministic)
    configure_gpu_memory_growth()
    image_size = _parse_image_size(args.image_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir_values = args.data_dir or [str(DEFAULT_DATA_DIR)]
    datasets = load_labeled_datasets(
        data_dirs=[Path(value).expanduser() for value in data_dir_values],
        representative_dir=Path(args.representative_dir).expanduser(),
        image_size=image_size,
        batch_size=args.batch_size,
        seed=args.split_seed,
        augment=True,
        validation_fraction=args.validation_fraction,
    )
    teacher = build_classifier(
        architecture="teacher",
        input_shape=(*image_size, 3),
        num_classes=len(EXPECTED_CLASSES),
        backbone_weights=None,
    )
    load_classifier_weights(teacher, Path(args.teacher).expanduser(), "teacher")
    teacher.trainable = False
    for layer in teacher.layers:
        layer.trainable = False
    teacher_report = evaluate_keras_model(teacher, datasets.validation)
    teacher_real_capture_report = evaluate_keras_model(
        teacher, datasets.representative
    )

    initial_specs = candidate_specs()[: args.initial_candidates]
    reports: list[dict[str, object]] = []
    for spec in initial_specs:
        reports.append(
            _train_candidate(
                spec=spec,
                seed=args.train_seed,
                teacher=teacher,
                datasets=datasets,
                output_dir=output_dir,
                image_size=image_size,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                representative_samples=args.representative_samples,
            )
        )
    preliminary_winner = max(reports, key=_float_rank_key)
    winner_name = preliminary_winner["candidate"]["name"]
    winner_spec = next(spec for spec in candidate_specs() if spec.name == winner_name)
    reports.append(
        _train_candidate(
            spec=CandidateSpec(
                name=f"refit_seed{args.refit_seed}_{winner_spec.name}",
                temperature=winner_spec.temperature,
                hard_loss_weight=winner_spec.hard_loss_weight,
                distillation=winner_spec.distillation,
            ),
            seed=args.refit_seed,
            teacher=teacher,
            datasets=datasets,
            output_dir=output_dir,
            image_size=image_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            representative_samples=args.representative_samples,
        )
    )
    ranked = _convert_ranked_candidates(
        reports,
        converter=Path(args.converter),
        wine_prefix=Path(args.wine_prefix),
        max_scratch=args.max_scratch,
        max_model_bytes=args.max_model_bytes,
        converter_timeout=args.converter_timeout,
        skip_neutron=args.skip_neutron,
    )
    selected = (
        ranked[0]
        if args.skip_neutron
        else next(
            (report for report in ranked if (report.get("neutron") or {}).get("passed")),
            None,
        )
    )
    search_report = {
        "config": {
            "teacher": str(Path(args.teacher).expanduser()),
            "data_dirs": [str(Path(value).expanduser()) for value in data_dir_values],
            "representative_dir": str(Path(args.representative_dir).expanduser()),
            "image_size": list(image_size),
            "batch_size": args.batch_size,
            "validation_fraction": args.validation_fraction,
            "split_seed": args.split_seed,
            "train_seed": args.train_seed,
            "refit_seed": args.refit_seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "representative_samples": args.representative_samples,
            "converter": str(Path(args.converter).expanduser()),
            "wine_prefix": str(Path(args.wine_prefix).expanduser()),
            "converter_timeout": args.converter_timeout,
            "max_scratch": args.max_scratch,
            "max_model_bytes": args.max_model_bytes,
        },
        "teacher_validation": teacher_report,
        "teacher_real_capture": teacher_real_capture_report,
        "train_counts": list(datasets.train_counts),
        "validation_counts": list(datasets.validation_counts),
        "ranking": [report["candidate"] for report in ranked],
        "candidates": ranked,
        "selected": selected["candidate"] if selected else None,
        "status": (
            "skipped_neutron_gate"
            if args.skip_neutron
            else "passed"
            if selected
            else "no_candidate_passed_neutron_gate"
        ),
    }
    write_json(output_dir / "search_report.json", search_report)
    if selected is None:
        raise RuntimeError(
            "no INT8-ranked student candidate passed the Neutron memory/model gate; "
            f"see {output_dir / 'search_report.json'}"
        )

    selected_dir = Path(selected["checkpoint"]).parent
    filenames = ["best.weights.h5", "box_student_int8.tflite"]
    if not args.skip_neutron:
        filenames.append("box_student_npu.tflite")
    for filename in filenames:
        source = selected_dir / filename
        if not source.is_file():
            raise RuntimeError(f"selected candidate is missing {source}")
        shutil.copy2(source, output_dir / filename)
    selected_report = dict(selected)
    selected_report["selected"] = True
    write_json(output_dir / "report.json", selected_report)
    print(json.dumps(search_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
