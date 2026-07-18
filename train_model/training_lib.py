"""Shared supervised-training utilities for the MCXVision classifiers."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


EXPECTED_CLASSES = tuple(f"{index:02d}" for index in range(10))
DEFAULT_IMAGE_SIZE = (120, 120)


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    representative: tf.data.Dataset
    validation: tf.data.Dataset
    class_names: tuple[str, ...]


def set_reproducibility(seed: int, deterministic: bool = False) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def configure_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # TensorFlow rejects this after the GPU runtime has been initialized.
            pass


def _directory_dataset(
    directory: Path,
    subset: str,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tuple[str, ...]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {directory}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        validation_split=0.2,
        subset=subset,
        seed=seed,
        shuffle=subset == "training",
        color_mode="rgb",
        image_size=image_size,
        interpolation="nearest",
        batch_size=batch_size,
        label_mode="int",
    )
    class_names = tuple(dataset.class_names)
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            f"unexpected classes in {directory}: {class_names}; "
            f"expected {EXPECTED_CLASSES}"
        )
    return dataset, class_names


def _concatenate(datasets: Sequence[tf.data.Dataset]) -> tf.data.Dataset:
    if not datasets:
        raise ValueError("at least one labeled dataset directory is required")
    result = datasets[0]
    for dataset in datasets[1:]:
        result = result.concatenate(dataset)
    return result


def _make_augmentation(seed: int) -> tf.keras.Sequential:
    # Keep augmentation out of the inference graph. Rotation and flips are not
    # used because they can change the meaning of visually similar classes.
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomZoom(
                height_factor=(-0.10, 0.10),
                width_factor=(-0.10, 0.10),
                fill_mode="constant",
                fill_value=0.0,
                seed=seed,
            )
        ],
        name="train_only_augmentation",
    )


def load_representative_dataset(
    directory: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    seed: int = 123,
    prefetch: int = 4,
) -> tf.data.Dataset:
    """Load all real captures in shuffled order for INT8 calibration."""

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"representative dataset directory does not exist: {directory}"
        )
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        seed=seed,
        shuffle=True,
        color_mode="rgb",
        image_size=image_size,
        interpolation="nearest",
        batch_size=batch_size,
        label_mode="int",
    )
    class_names = tuple(dataset.class_names)
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            f"unexpected classes in representative dataset {directory}: "
            f"{class_names}; expected {EXPECTED_CLASSES}"
        )
    return dataset.prefetch(prefetch)


def load_labeled_datasets(
    data_dirs: Sequence[Path],
    representative_dir: Path | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    seed: int = 123,
    augment: bool = True,
    prefetch: int = 4,
) -> DatasetBundle:
    """Load every directory as labeled data and combine matching splits."""

    train_parts: list[tf.data.Dataset] = []
    validation_parts: list[tf.data.Dataset] = []
    for data_dir in data_dirs:
        train_part, class_names = _directory_dataset(
            Path(data_dir), "training", image_size, batch_size, seed
        )
        validation_part, validation_names = _directory_dataset(
            Path(data_dir), "validation", image_size, batch_size, seed
        )
        if validation_names != class_names:
            raise ValueError(f"class order changed between splits in {data_dir}")
        train_parts.append(train_part)
        validation_parts.append(validation_part)

    train_raw = _concatenate(train_parts)
    validation = _concatenate(validation_parts)

    # Shuffle batches after concatenation so one source directory does not
    # dominate a contiguous part of each epoch.
    train_raw = train_raw.shuffle(
        buffer_size=max(16, 16 * len(train_parts)),
        seed=seed,
        reshuffle_each_iteration=True,
    )
    representative = (
        load_representative_dataset(
            representative_dir,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            prefetch=prefetch,
        )
        if representative_dir is not None
        else train_raw.prefetch(prefetch)
    )

    if augment:
        augmentation = _make_augmentation(seed)

        def apply_augmentation(images, labels):
            return augmentation(images, training=True), labels

        train = train_raw.map(
            apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        train = train_raw

    return DatasetBundle(
        train=train.prefetch(prefetch),
        representative=representative,
        validation=validation.prefetch(prefetch),
        class_names=EXPECTED_CLASSES,
    )


def load_test_dataset(
    directory: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    prefetch: int = 4,
) -> tf.data.Dataset:
    if not directory.is_dir():
        raise FileNotFoundError(f"test directory does not exist: {directory}")
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        shuffle=False,
        class_names=list(EXPECTED_CLASSES),
        color_mode="rgb",
        image_size=image_size,
        interpolation="nearest",
        batch_size=batch_size,
        label_mode="int",
    )
    return dataset.prefetch(prefetch)


def _conv_block(
    inputs: tf.Tensor,
    channels: int,
    block_name: str,
    stride: int,
) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        channels,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{block_name}_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn")(x)
    return tf.keras.layers.ReLU(name=f"{block_name}_relu")(x)


def build_classifier(
    architecture: str = "student",
    input_shape: tuple[int, int, int] = (120, 120, 3),
    num_classes: int = 10,
    dropout: float | None = None,
    backbone_weights: str | None = None,
    mobilenet_alpha: float = 1.0,
) -> tf.keras.Model:
    """Build the MobileNetV2 teacher or deployable MCX student."""

    if dropout is None:
        dropout = 0.20 if architecture == "teacher" else 0.10
    inputs = tf.keras.Input(shape=input_shape, name="image")
    if architecture == "teacher":
        if min(input_shape[:2]) < 32:
            raise ValueError("MobileNetV2 requires image dimensions of at least 32")
        # Keep the public model interface at raw RGB 0..255 so OpenART can pass
        # camera pixels directly. Full INT8 conversion quantizes this operation.
        x = tf.keras.layers.Rescaling(
            scale=1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess"
        )(inputs)
        try:
            backbone = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                alpha=mobilenet_alpha,
                include_top=False,
                weights=backbone_weights,
            )
        except Exception as error:
            if backbone_weights is None:
                raise
            raise RuntimeError(
                "failed to load MobileNetV2 pretrained weights. Check network "
                "access, the local no-top weights path, and mobilenet alpha; "
                "use --backbone-weights none for random initialization"
            ) from error
        x = backbone(x)
    elif architecture == "student":
        if backbone_weights is not None:
            raise ValueError("the student architecture has no pretrained backbone")
        x = inputs
        for index, channels in enumerate((16, 32, 64, 128), start=1):
            x = _conv_block(x, channels, f"block{index}", stride=2)
        x = _conv_block(x, 128, "tail1", stride=1)
        x = _conv_block(x, 128, "tail2", stride=1)
    else:
        raise ValueError(f"unknown architecture: {architecture}")

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(x)
    logits = tf.keras.layers.Dense(num_classes, name="classifier")(x)
    return tf.keras.Model(inputs, logits, name=f"box_classifier_{architecture}")


def load_classifier_weights(
    model: tf.keras.Model, checkpoint: Path, architecture: str
) -> str:
    """Strictly load weights produced for the same teacher/student graph."""

    checkpoint = Path(checkpoint)
    try:
        model.load_weights(str(checkpoint))
        return "current"
    except ValueError as direct_error:
        raise ValueError(
            f"checkpoint is incompatible with the {architecture} architecture: "
            f"{checkpoint}. The old three-layer model cannot initialize the "
            "structurally different student; transfer it by distillation instead."
        ) from direct_error


def architecture_warning(architecture: str) -> str | None:
    if architecture == "teacher":
        return (
            "The MobileNetV2 teacher targets OpenART and is not subject to the "
            "MCXN947 Neutron limits. Run its final TFLite model on real OpenART "
            "hardware before using it for distillation."
        )
    return None


@tf.keras.utils.register_keras_serializable(package="mcxvision")
class SparseCrossentropyWithLabelSmoothing(tf.keras.losses.Loss):
    """Sparse-label cross entropy with configurable label smoothing."""

    def __init__(
        self,
        label_smoothing: float = 0.10,
        from_logits: bool = True,
        name: str = "smoothed_sparse_crossentropy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        return tf.keras.losses.categorical_crossentropy(
            one_hot,
            y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "label_smoothing": self.label_smoothing,
                "from_logits": self.from_logits,
            }
        )
        return config


def compile_classifier(
    model: tf.keras.Model,
    learning_rate,
    weight_decay: float,
    label_smoothing: float,
) -> None:
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    if hasattr(optimizer, "exclude_from_weight_decay"):
        optimizer.exclude_from_weight_decay(var_names=["bias", "beta", "gamma"])
    model.compile(
        optimizer=optimizer,
        loss=SparseCrossentropyWithLabelSmoothing(
            label_smoothing=label_smoothing, from_logits=True
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def _predictions_from_keras(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for images, batch_labels in dataset:
        logits = model(images, training=False)
        labels.append(np.asarray(batch_labels).reshape(-1))
        predictions.append(np.argmax(np.asarray(logits), axis=-1).reshape(-1))
    return np.concatenate(labels), np.concatenate(predictions)


def classification_report(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=range(len(EXPECTED_CLASSES)))
    supports = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        supports,
        out=np.zeros(len(EXPECTED_CLASSES), dtype=np.float64),
        where=supports != 0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=range(len(EXPECTED_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "correct": int(np.sum(labels == predictions)),
        "total": int(labels.size),
        "per_class_recall": recalls.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_keras_model(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> dict[str, object]:
    labels, predictions = _predictions_from_keras(model, dataset)
    return classification_report(labels, predictions)


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Publish val_macro_f1 and save only genuine improvements."""

    def __init__(
        self,
        validation: tf.data.Dataset,
        filepath: Path,
        initial_best: float = -1.0,
        min_delta: float = 1e-5,
    ):
        super().__init__()
        self.validation = validation
        self.filepath = Path(filepath)
        self.best_f1 = float(initial_best)
        self.min_delta = float(min_delta)
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Preserve the incoming baseline. This matters during fine-tuning when
        # an entire phase can be worse than the pretrained model.
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        report = evaluate_keras_model(self.model, self.validation)
        current_f1 = float(report["macro_f1"])
        if logs is not None:
            logs["val_macro_f1"] = current_f1
        if current_f1 > self.best_f1 + self.min_delta:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.best_weights = self.model.get_weights()
            self.model.save_weights(str(self.filepath))
            self.best_f1 = current_f1
            print(f"val_macro_f1 improved to {current_f1:.6f}; checkpoint saved")

    def on_train_end(self, logs=None):
        if self.best_weights is None:
            return
        self.model.set_weights(self.best_weights)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.filepath))


def make_callbacks(
    validation: tf.data.Dataset,
    checkpoint: Path,
    patience: int,
    initial_best: float = -1.0,
) -> tuple[list[tf.keras.callbacks.Callback], MacroF1Checkpoint]:
    macro_f1 = MacroF1Checkpoint(
        validation=validation,
        filepath=checkpoint,
        initial_best=initial_best,
    )
    callbacks: list[tf.keras.callbacks.Callback] = [
        macro_f1,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=patience,
            baseline=initial_best if initial_best >= 0.0 else None,
            restore_best_weights=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return callbacks, macro_f1


def fit_training_phase(
    *,
    name: str,
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    checkpoint: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    patience: int,
    initial_best: float = -1.0,
    schedule_alpha: float = 0.10,
) -> tuple[dict[str, list[float]], float]:
    """Fit one stage and restore the best global macro-F1 weights."""

    if epochs <= 0:
        return {}, initial_best
    steps = int(tf.data.experimental.cardinality(train_dataset).numpy())
    if steps <= 0:
        raise RuntimeError("training dataset cardinality must be finite and non-empty")
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=max(1, steps * epochs),
        alpha=schedule_alpha,
    )
    compile_classifier(
        model,
        learning_rate=schedule,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
    )
    callbacks, macro_f1 = make_callbacks(
        validation=validation_dataset,
        checkpoint=checkpoint,
        patience=min(patience, max(1, epochs)),
        initial_best=initial_best,
    )
    print(f"\n===== {name} =====")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    return (
        {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        max(initial_best, macro_f1.best_f1),
    )


def set_finetune_blocks(
    model: tf.keras.Model, trainable_tail_blocks: int
) -> None:
    """Freeze BN and train the classifier plus the requested final blocks."""

    mobilenet_backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower()
    ]
    if mobilenet_backbones:
        backbone = mobilenet_backbones[0]
        for layer in model.layers:
            layer.trainable = layer.name == "classifier"
        if trainable_tail_blocks <= 0:
            backbone.trainable = False
            return

        ordered_groups = ["stem", "block_0"] + [
            f"block_{index}" for index in range(1, 17)
        ] + ["top"]
        selected = set(ordered_groups[-trainable_tail_blocks:])

        def mobile_group(name: str) -> str:
            if name.startswith("block_"):
                parts = name.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    return f"block_{parts[1]}"
            if name.startswith("expanded_conv"):
                return "block_0"
            if name.startswith("Conv_1") or name == "out_relu":
                return "top"
            return "stem"

        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = (
                mobile_group(layer.name) in selected
                and not isinstance(layer, tf.keras.layers.BatchNormalization)
            )
        return

    block_names: list[str] = []
    for layer in model.layers:
        if layer.name.endswith("_conv"):
            block_names.append(layer.name.removesuffix("_conv"))
    selected = (
        set(block_names[-trainable_tail_blocks:])
        if trainable_tail_blocks > 0
        else set()
    )

    for layer in model.layers:
        if layer.name == "classifier":
            layer.trainable = True
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        elif any(layer.name.startswith(f"{name}_") for name in selected):
            layer.trainable = True
        else:
            layer.trainable = False


def export_int8_model(
    model: tf.keras.Model,
    representative_dataset: tf.data.Dataset,
    output_path: Path,
    representative_samples: int = 500,
) -> dict[str, object]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = tf.keras.Input(shape=model.input_shape[1:], name="image")
    probabilities = tf.keras.layers.Softmax(name="probabilities")(
        model(inputs, training=False)
    )
    probability_model = tf.keras.Model(inputs, probabilities)

    def representative_data():
        emitted = 0
        for images, _ in representative_dataset:
            for image in images:
                yield [tf.expand_dims(tf.cast(image, tf.float32), axis=0)]
                emitted += 1
                if emitted >= representative_samples:
                    return

    converter = tf.lite.TFLiteConverter.from_keras_model(probability_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    output_path.write_bytes(model_bytes)

    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError("exported model is not full INT8 at its public interface")
    return {
        "path": str(output_path),
        "size_bytes": len(model_bytes),
        "input_shape": input_detail["shape"].tolist(),
        "input_quantization": list(input_detail["quantization"]),
        "output_shape": output_detail["shape"].tolist(),
        "output_quantization": list(output_detail["quantization"]),
    }


def evaluate_tflite_model(
    model_path: Path, dataset: tf.data.Dataset
) -> dict[str, object]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    if input_detail["dtype"] != np.int8 or input_scale <= 0.0:
        raise RuntimeError("expected an int8 input tensor with valid quantization")

    labels: list[int] = []
    predictions: list[int] = []
    for images, batch_labels in dataset:
        for image, label in zip(np.asarray(images), np.asarray(batch_labels)):
            quantized = np.clip(
                np.rint(image / input_scale + input_zero), -128, 127
            ).astype(np.int8)
            interpreter.set_tensor(
                input_detail["index"], np.expand_dims(quantized, axis=0)
            )
            interpreter.invoke()
            output = interpreter.get_tensor(output_detail["index"])
            labels.append(int(label))
            predictions.append(int(np.argmax(output)))
    report = classification_report(np.asarray(labels), np.asarray(predictions))
    report.update(
        {
            "model": str(model_path),
            "input_quantization": list(input_detail["quantization"]),
            "output_quantization": list(output_detail["quantization"]),
        }
    )
    return report


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_image_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("image size must look like 120x120")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return height, width


def default_data_dirs(
    values: Iterable[str] | None, defaults: Sequence[Path]
) -> list[Path]:
    if values:
        return [Path(value).expanduser() for value in values]
    return [Path(value).expanduser() for value in defaults]
