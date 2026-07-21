"""Shared supervised-training utilities for the MCXVision classifiers."""

from __future__ import annotations

import io
import json
import math
import os
import random
import re
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


EXPECTED_CLASSES = tuple(f"{index:02d}" for index in range(10))
DEFAULT_IMAGE_SIZE = (120, 120)
_RESOURCE_TENSOR_SPEC = re.compile(
    r"^\s*\d+:\s*TensorSpec\(shape=\(\),\s*"
    r"dtype=tf\.resource,\s*name=None\)\s*$"
)


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    representative: tf.data.Dataset
    validation: tf.data.Dataset
    class_names: tuple[str, ...]
    train_counts: tuple[int, ...]
    validation_counts: tuple[int, ...]


@dataclass
class OpenArtDatasetBundle:
    train: tf.data.Dataset
    representative: tf.data.Dataset
    validation: tf.data.Dataset
    class_names: tuple[str, ...]
    train_counts: tuple[int, ...]
    validation_counts: tuple[int, ...]
    train_real_counts: tuple[int, ...]
    train_shotmix_counts: tuple[int, ...]
    validation_real_counts: tuple[int, ...]


@contextmanager
def _hide_resource_tensor_specs():
    """Filter TensorFlow converter's internal resource-handle dump."""

    captured = ((sys.stdout, io.StringIO()), (sys.stderr, io.StringIO()))
    try:
        with redirect_stdout(captured[0][1]), redirect_stderr(captured[1][1]):
            yield
    finally:
        for original, buffer in captured:
            for line in buffer.getvalue().splitlines(keepends=True):
                if not _RESOURCE_TENSOR_SPEC.fullmatch(line.rstrip()):
                    original.write(line)


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


_IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png"})


def _stratified_file_split(
    directory: Path,
    validation_fraction: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {directory}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be between 0 and 1")

    train_paths: dict[str, list[str]] = {}
    validation_paths: dict[str, list[str]] = {}
    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        class_dir = directory / class_name
        if not class_dir.is_dir():
            raise ValueError(f"missing class directory {class_dir}")
        paths = sorted(
            str(path)
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"class directory is empty: {class_dir}")
        # Shuffle within each class before slicing so validation remains
        # stratified instead of inheriting the global class-directory order.
        random.Random(seed + class_index).shuffle(paths)
        validation_count = (
            min(len(paths) - 1, max(1, int(math.ceil(len(paths) * validation_fraction))))
            if len(paths) > 1
            else 0
        )
        validation_paths[class_name] = paths[:validation_count]
        train_paths[class_name] = paths[validation_count:]
    return train_paths, validation_paths


def _paths_dataset(
    paths_by_class: dict[str, list[str]],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    paths: list[str] = []
    labels: list[int] = []
    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        class_paths = paths_by_class[class_name]
        paths.extend(class_paths)
        labels.extend([class_index] * len(class_paths))
    if not paths:
        raise ValueError("dataset split is empty")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode_and_resize(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size, method="nearest")
        return image, label

    dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(len(paths), batch_size * 4),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size)


def _directory_dataset(
    directory: Path,
    subset: str,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
    validation_fraction: float = 0.20,
) -> tuple[tf.data.Dataset, tuple[str, ...], tuple[int, ...]]:
    train_paths, validation_paths = _stratified_file_split(
        directory, validation_fraction=validation_fraction, seed=seed
    )
    if subset == "training":
        paths_by_class = train_paths
        shuffle = True
    elif subset == "validation":
        paths_by_class = validation_paths
        shuffle = False
    else:
        raise ValueError(f"unknown dataset subset: {subset}")

    dataset = _paths_dataset(
        paths_by_class,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    counts = tuple(len(paths_by_class[name]) for name in EXPECTED_CLASSES)
    return dataset, EXPECTED_CLASSES, counts


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
    validation_fraction: float = 0.20,
    prefetch: int = 4,
) -> DatasetBundle:
    """Load every directory with a reproducible, per-class random split."""

    train_parts: list[tf.data.Dataset] = []
    validation_parts: list[tf.data.Dataset] = []
    train_counts: list[int] = [0] * len(EXPECTED_CLASSES)
    validation_counts: list[int] = [0] * len(EXPECTED_CLASSES)
    for data_dir in data_dirs:
        train_part, class_names, train_count = _directory_dataset(
            Path(data_dir),
            "training",
            image_size,
            batch_size,
            seed,
            validation_fraction,
        )
        validation_part, validation_names, validation_count = _directory_dataset(
            Path(data_dir),
            "validation",
            image_size,
            batch_size,
            seed,
            validation_fraction,
        )
        if validation_names != class_names:
            raise ValueError(f"class order changed between splits in {data_dir}")
        train_parts.append(train_part)
        validation_parts.append(validation_part)
        train_counts = [left + right for left, right in zip(train_counts, train_count)]
        validation_counts = [
            left + right for left, right in zip(validation_counts, validation_count)
        ]

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
        train_counts=tuple(train_counts),
        validation_counts=tuple(validation_counts),
    )


def _openart_augmentation(images: tf.Tensor) -> tf.Tensor:
    """Apply only the mild photometric changes allowed for real captures."""

    images = tf.cast(images, tf.float32)
    batch_size = tf.shape(images)[0]

    # 这里复刻 ShotMix 前景的轻微亮度/对比度范围；增强只挂在实拍训练流，
    # 不进入推理图，也不对验证集或代表性校准数据生效。
    brightness_contrast_mask = tf.random.uniform([batch_size, 1, 1, 1]) < 0.30
    brightness = tf.random.uniform(
        [batch_size, 1, 1, 1], minval=-0.08 * 255.0, maxval=0.08 * 255.0
    )
    contrast = tf.random.uniform([batch_size, 1, 1, 1], minval=0.92, maxval=1.08)
    mean = tf.reduce_mean(images, axis=[1, 2], keepdims=True)
    brightness_contrast = (images - mean) * contrast + mean + brightness
    images = tf.where(
        brightness_contrast_mask,
        tf.clip_by_value(brightness_contrast, 0.0, 255.0),
        images,
    )

    # 颜色变化只选择 RGB shift 或 HSV shift 之一，明确不加入模糊和噪声。
    color_mask = tf.random.uniform([batch_size, 1, 1, 1]) < 0.20
    use_rgb_shift = tf.random.uniform([batch_size, 1, 1, 1]) < 0.5
    rgb_shift = images + tf.random.uniform(
        [batch_size, 1, 1, 3],
        minval=tf.constant([-3.0, -3.0, -4.0]),
        maxval=tf.constant([4.0, 3.0, 4.0]),
    )
    hsv = tf.image.rgb_to_hsv(tf.clip_by_value(images / 255.0, 0.0, 1.0))
    hsv_delta = tf.concat(
        [
            tf.random.uniform([batch_size, 1, 1, 1], -2.0 / 360.0, 2.0 / 360.0),
            tf.random.uniform([batch_size, 1, 1, 1], -4.0 / 255.0, 4.0 / 255.0),
            tf.random.uniform([batch_size, 1, 1, 1], -3.0 / 255.0, 3.0 / 255.0),
        ],
        axis=-1,
    )
    hsv_shift = tf.image.hsv_to_rgb(tf.clip_by_value(hsv + hsv_delta, 0.0, 1.0))
    hsv_shift = hsv_shift * 255.0
    color_shift = tf.where(use_rgb_shift, rgb_shift, hsv_shift)
    return tf.where(
        color_mask,
        tf.clip_by_value(color_shift, 0.0, 255.0),
        images,
    )


def _openart_source_split(
    directory: Path,
    validation_fraction: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {directory}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be between 0 and 1")

    train_real: dict[str, list[str]] = {}
    train_shotmix: dict[str, list[str]] = {}
    validation_real: dict[str, list[str]] = {}
    # 仅从原始实拍中留出验证集；所有 shotmix_ 文件必须留在训练集，
    # 这样验证 macro-F1 才能反映真实摄像头域，而不是合成图域。
    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        class_dir = directory / class_name
        if not class_dir.is_dir():
            raise ValueError(f"missing class directory {class_dir}")
        paths = sorted(
            str(path)
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        real_paths = [path for path in paths if not Path(path).name.startswith("shotmix_")]
        shotmix_paths = [path for path in paths if Path(path).name.startswith("shotmix_")]
        if len(real_paths) < 2:
            raise ValueError(
                f"class {class_name} needs at least two original real images, "
                f"found {len(real_paths)}"
            )
        random.Random(seed + class_index).shuffle(real_paths)
        validation_count = min(
            len(real_paths) - 1,
            max(1, int(math.ceil(len(real_paths) * validation_fraction))),
        )
        validation_real[class_name] = real_paths[:validation_count]
        train_real[class_name] = real_paths[validation_count:]
        train_shotmix[class_name] = shotmix_paths
    return train_real, train_shotmix, validation_real


def _paths_dataset_from_classes(
    paths_by_class: dict[str, list[str]],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = _paths_dataset(
        paths_by_class,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def load_openart_datasets(
    directory: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    seed: int = 123,
    validation_fraction: float = 0.20,
    prefetch: int = 4,
) -> OpenArtDatasetBundle:
    """Split real captures for validation and train all ShotMix samples."""

    train_real_paths, train_shotmix_paths, validation_paths = _openart_source_split(
        Path(directory), validation_fraction=validation_fraction, seed=seed
    )
    train_real = _paths_dataset_from_classes(
        train_real_paths, image_size, batch_size, shuffle=True, seed=seed
    )
    # 两条训练流分别保留来源：只有原始实拍流使用轻度在线增强。
    train_real = train_real.map(
        lambda images, labels: (_openart_augmentation(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train = train_real
    if any(train_shotmix_paths[name] for name in EXPECTED_CLASSES):
        train_shotmix = _paths_dataset_from_classes(
            train_shotmix_paths, image_size, batch_size, shuffle=True, seed=seed + 1000
        )
        train = train.concatenate(train_shotmix)
    train = train.shuffle(
        buffer_size=max(
            16,
            sum(len(paths) for paths in train_real_paths.values())
            + sum(len(paths) for paths in train_shotmix_paths.values()),
        ),
        seed=seed,
        reshuffle_each_iteration=True,
    )
    validation = _paths_dataset_from_classes(
        validation_paths, image_size, batch_size, shuffle=False, seed=seed
    )
    # 校准只看训练部分的原始实拍，避免把验证图和合成退化带入量化范围。
    representative = _paths_dataset_from_classes(
        train_real_paths, image_size, batch_size, shuffle=True, seed=seed + 2000
    )

    train_real_counts = tuple(len(train_real_paths[name]) for name in EXPECTED_CLASSES)
    train_shotmix_counts = tuple(
        len(train_shotmix_paths[name]) for name in EXPECTED_CLASSES
    )
    validation_real_counts = tuple(
        len(validation_paths[name]) for name in EXPECTED_CLASSES
    )
    train_counts = tuple(
        real + shotmix
        for real, shotmix in zip(train_real_counts, train_shotmix_counts)
    )
    return OpenArtDatasetBundle(
        train=train.prefetch(prefetch),
        representative=representative.prefetch(prefetch),
        validation=validation.prefetch(prefetch),
        class_names=EXPECTED_CLASSES,
        train_counts=train_counts,
        validation_counts=validation_real_counts,
        train_real_counts=train_real_counts,
        train_shotmix_counts=train_shotmix_counts,
        validation_real_counts=validation_real_counts,
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


OPENART_CNN_CONFIGS: dict[str, dict[str, object]] = {
    "narrow": {"channels": (16, 32, 64, 96), "dropout": 0.30},
    "medium": {"channels": (24, 48, 96, 128), "dropout": 0.25},
    "wide": {"channels": (32, 64, 128, 160), "dropout": 0.20},
}


def build_openart_classifier(
    variant: str,
    input_shape: tuple[int, int, int] = (120, 120, 3),
    num_classes: int = 10,
    weight_decay: float = 1e-4,
) -> tf.keras.Model:
    """Build a raw-RGB, builtin-op CNN intended for OpenART deployment."""

    if variant not in OPENART_CNN_CONFIGS:
        raise ValueError(
            f"unknown OpenART CNN variant {variant!r}; "
            f"choose from {tuple(OPENART_CNN_CONFIGS)}"
        )
    config = OPENART_CNN_CONFIGS[variant]
    channels = tuple(int(value) for value in config["channels"])
    dropout = float(config["dropout"])
    regularizer = tf.keras.regularizers.l2(weight_decay)

    # 保持公共接口为 [0,255] RGB；Rescaling 在量化图内部完成归一化。
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="openart_preprocess")(inputs)
    # 四个 stride=2 block 将 120x120 降到 8x8，再用一个 stride=1 tail
    # 提取局部纹理；只使用 OpenART 已验证的 builtin 算子。
    for index, channel_count in enumerate(channels, start=1):
        x = tf.keras.layers.Conv2D(
            channel_count,
            3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizer,
            name=f"openart_block{index}_conv",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"openart_block{index}_bn")(x)
        x = tf.keras.layers.ReLU(name=f"openart_block{index}_relu")(x)
    x = tf.keras.layers.Conv2D(
        channels[-1],
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name="openart_tail_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="openart_tail_bn")(x)
    x = tf.keras.layers.ReLU(name="openart_tail_relu")(x)
    # GAP 代替大尺寸全连接层，降低参数量和过拟合风险。
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(x)
    logits = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=regularizer,
        name="classifier",
    )(x)
    return tf.keras.Model(inputs, logits, name=f"box_classifier_openart_{variant}")


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

    with _hide_resource_tensor_specs():
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
