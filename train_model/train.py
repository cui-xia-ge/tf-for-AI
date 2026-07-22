"""Train and select an INT8 box classifier with disjoint real-image splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


CLASSES = tuple(f"{index:02d}" for index in range(10))
IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png"})
IMAGE_SIZE = (120, 120)
DEFAULT_TOTAL_DIR = Path("/home/cgcgs/718/dataset/box/total")
DEFAULT_REAL_DIR = Path("/home/cgcgs/718/dataset/box/real")
DEFAULT_OUTPUT_DIR = Path("artifacts/total_real_mmd")
SPLIT_NAMES = ("train", "validation", "mmd", "test")

# real 训练图只使用 ShotMix 前景增强中允许的轻微颜色变化。
REAL_AUGMENTATION_SPEC = {
    "brightness_contrast_probability": 0.30,
    "brightness_limit": 0.08,
    "contrast_limit": 0.08,
    "color_probability": 0.20,
    "rgb_shift_limits": ((-3.0, 4.0), (-3.0, 3.0), (-4.0, 4.0)),
    "hue_shift_limit": 2.0,
    "saturation_shift_limit": 4.0,
    "value_shift_limit": 3.0,
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_name: str
    label: int
    source: str
    sha256: str
    group_id: str = ""


@dataclass(frozen=True)
class CandidateCheckpoint:
    epoch: int
    macro_f1: float
    path: Path


def configure_tensorflow(seed: int) -> list[str]:
    """Configure deterministic seeds and GPU memory growth before model creation."""

    tf.keras.utils.set_random_seed(seed)
    gpu_names: list[str] = []
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # 必须按需申请显存，避免训练启动时占满整块 GPU。
        tf.config.experimental.set_memory_growth(gpu, True)
        details = tf.config.experimental.get_device_details(gpu)
        gpu_names.append(str(details.get("device_name", gpu.name)))

    print(f"Python: {sys.executable}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {gpus}")
    return gpu_names


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_records(
    root: Path,
    source: str,
    max_per_class: int = 0,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for label, class_name in enumerate(CLASSES):
        paths = image_paths(root / class_name)
        if max_per_class > 0:
            paths = paths[:max_per_class]
        if not paths:
            raise ValueError(f"类别 {class_name} 没有可用图片: {root / class_name}")
        for path in paths:
            records.append(
                ImageRecord(
                    path=path.resolve(),
                    class_name=class_name,
                    label=label,
                    source=source,
                    sha256=sha256_file(path),
                )
            )
    return records


def normalize_capture_stem(path: Path) -> str:
    """Treat names such as ``35.jpg`` and ``35 (2).jpg`` as one capture group."""

    return re.sub(r" \(\d+\)$", "", path.stem).casefold()


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, index: int) -> int:
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


def group_real_records(records: Sequence[ImageRecord]) -> list[list[ImageRecord]]:
    """Group exact copies and filename variants so they cannot cross splits."""

    grouped_by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped_by_class[record.class_name].append(record)

    result: list[list[ImageRecord]] = []
    for class_name in CLASSES:
        class_records = sorted(grouped_by_class[class_name], key=lambda item: str(item.path))
        disjoint_set = _DisjointSet(len(class_records))
        first_by_stem: dict[str, int] = {}
        first_by_hash: dict[str, int] = {}
        for index, record in enumerate(class_records):
            stem = normalize_capture_stem(record.path)
            if stem in first_by_stem:
                disjoint_set.union(index, first_by_stem[stem])
            else:
                first_by_stem[stem] = index
            if record.sha256 in first_by_hash:
                disjoint_set.union(index, first_by_hash[record.sha256])
            else:
                first_by_hash[record.sha256] = index

        components: dict[int, list[ImageRecord]] = defaultdict(list)
        for index, record in enumerate(class_records):
            components[disjoint_set.find(index)].append(record)
        result.extend(components.values())
    return result


def _largest_remainder_counts(total: int, ratios: Sequence[float]) -> list[int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remaining = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def split_real_records(
    records: Sequence[ImageRecord],
    ratios: Sequence[float],
    seed: int,
) -> dict[str, list[ImageRecord]]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"real 划分比例必须有 {len(SPLIT_NAMES)} 项")
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("real 划分比例必须全部大于0且总和为1")

    groups_by_class: dict[str, list[list[ImageRecord]]] = defaultdict(list)
    for group in group_real_records(records):
        groups_by_class[group[0].class_name].append(group)

    splits: dict[str, list[ImageRecord]] = {name: [] for name in SPLIT_NAMES}
    for label, class_name in enumerate(CLASSES):
        groups = groups_by_class[class_name]
        if len(groups) < len(SPLIT_NAMES):
            raise ValueError(f"类别 {class_name} 的独立图片组不足以完成四路划分")
        random.Random(seed + label).shuffle(groups)
        group_counts = _largest_remainder_counts(len(groups), ratios)
        if any(count == 0 for count in group_counts):
            raise ValueError(f"类别 {class_name} 的某个 real 子集为空")

        cursor = 0
        for split_name, count in zip(SPLIT_NAMES, group_counts):
            selected_groups = groups[cursor : cursor + count]
            cursor += count
            for group in selected_groups:
                group_identity = hashlib.sha256(
                    "\n".join(sorted(str(item.path) for item in group)).encode("utf-8")
                ).hexdigest()[:16]
                splits[split_name].extend(
                    ImageRecord(
                        path=item.path,
                        class_name=item.class_name,
                        label=item.label,
                        source=item.source,
                        sha256=item.sha256,
                        group_id=f"{class_name}:{group_identity}",
                    )
                    for item in group
                )

    for split_records in splits.values():
        split_records.sort(key=lambda item: (item.label, str(item.path)))
    return splits


def audit_no_leakage(
    total_records: Sequence[ImageRecord],
    real_splits: dict[str, Sequence[ImageRecord]],
) -> None:
    """Fail before training if any real identity appears in two data roles."""

    path_sets: dict[str, set[Path]] = {
        name: {record.path for record in records}
        for name, records in real_splits.items()
    }
    hash_sets: dict[str, set[str]] = {
        name: {record.sha256 for record in records}
        for name, records in real_splits.items()
    }
    group_sets: dict[str, set[str]] = {
        name: {record.group_id for record in records}
        for name, records in real_splits.items()
    }
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            if path_sets[left_name] & path_sets[right_name]:
                raise ValueError(f"real 路径跨集合重复: {left_name}/{right_name}")
            if hash_sets[left_name] & hash_sets[right_name]:
                raise ValueError(f"real 文件哈希跨集合重复: {left_name}/{right_name}")
            if group_sets[left_name] & group_sets[right_name]:
                raise ValueError(f"real 近重复组跨集合重复: {left_name}/{right_name}")

    training_hashes = {record.sha256 for record in total_records} | hash_sets["train"]
    for held_out_name in ("validation", "mmd", "test"):
        if training_hashes & hash_sets[held_out_name]:
            raise ValueError(f"训练数据与 real-{held_out_name} 存在相同图片")

    for split_name, records in real_splits.items():
        present_classes = {record.class_name for record in records}
        if present_classes != set(CLASSES):
            raise ValueError(f"real-{split_name} 缺少类别: {set(CLASSES) - present_classes}")


def validate_image_shapes(records: Sequence[ImageRecord]) -> None:
    for record in records:
        encoded = tf.io.read_file(str(record.path))
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        shape = tuple(int(value) for value in tf.shape(image).numpy())
        if shape != (IMAGE_SIZE[0], IMAGE_SIZE[1], 3):
            raise ValueError(f"图片尺寸必须为120x120 RGB: {record.path} -> {shape}")


def count_by_class(records: Sequence[ImageRecord]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in CLASSES}
    for record in records:
        counts[record.class_name] += 1
    return counts


def write_split_manifest(
    path: Path,
    total_records: Sequence[ImageRecord],
    real_splits: dict[str, Sequence[ImageRecord]],
    ratios: Sequence[float],
    seed: int,
) -> None:
    records: list[dict[str, object]] = []
    for record in total_records:
        records.append(
            {
                **asdict(record),
                "path": str(record.path),
                "split": "train",
            }
        )
    for split_name, split_records in real_splits.items():
        for record in split_records:
            records.append(
                {
                    **asdict(record),
                    "path": str(record.path),
                    "split": split_name,
                }
            )

    payload = {
        "seed": seed,
        "classes": list(CLASSES),
        "real_ratios": dict(zip(SPLIT_NAMES, ratios)),
        "counts": {
            "total_train": count_by_class(total_records),
            **{
                f"real_{name}": count_by_class(split_records)
                for name, split_records in real_splits.items()
            },
        },
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def decode_record(path: tf.Tensor, label: tf.Tensor, source: tf.Tensor):
    encoded = tf.io.read_file(path)
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return tf.cast(image, tf.float32), label, source


def augment_real_image(image: tf.Tensor) -> tf.Tensor:
    """Apply only the color operations permitted for real training captures."""

    spec = REAL_AUGMENTATION_SPEC

    def brightness_contrast() -> tf.Tensor:
        brightness = tf.random.uniform(
            (), -spec["brightness_limit"] * 255.0, spec["brightness_limit"] * 255.0
        )
        contrast = tf.random.uniform(
            (), 1.0 - spec["contrast_limit"], 1.0 + spec["contrast_limit"]
        )
        mean = tf.reduce_mean(image, axis=(0, 1), keepdims=True)
        return (image - mean) * contrast + mean + brightness

    image = tf.cond(
        tf.random.uniform(()) < spec["brightness_contrast_probability"],
        brightness_contrast,
        lambda: image,
    )

    def rgb_shift() -> tf.Tensor:
        shifts = [
            tf.random.uniform((), lower, upper)
            for lower, upper in spec["rgb_shift_limits"]
        ]
        return image + tf.reshape(tf.stack(shifts), (1, 1, 3))

    def hsv_shift() -> tf.Tensor:
        hsv = tf.image.rgb_to_hsv(tf.clip_by_value(image, 0.0, 255.0) / 255.0)
        hue = tf.math.floormod(
            hsv[..., 0]
            + tf.random.uniform(
                (), -spec["hue_shift_limit"] / 180.0, spec["hue_shift_limit"] / 180.0
            ),
            1.0,
        )
        saturation = tf.clip_by_value(
            hsv[..., 1]
            + tf.random.uniform(
                (),
                -spec["saturation_shift_limit"] / 255.0,
                spec["saturation_shift_limit"] / 255.0,
            ),
            0.0,
            1.0,
        )
        value = tf.clip_by_value(
            hsv[..., 2]
            + tf.random.uniform(
                (),
                -spec["value_shift_limit"] / 255.0,
                spec["value_shift_limit"] / 255.0,
            ),
            0.0,
            1.0,
        )
        return tf.image.hsv_to_rgb(tf.stack((hue, saturation, value), axis=-1)) * 255.0

    image = tf.cond(
        tf.random.uniform(()) < spec["color_probability"],
        lambda: tf.cond(tf.random.uniform(()) < 0.5, rgb_shift, hsv_shift),
        lambda: image,
    )
    return tf.clip_by_value(image, 0.0, 255.0)


def make_source_dataset(
    total_records: Sequence[ImageRecord],
    real_train_records: Sequence[ImageRecord],
    batch_size: int,
    seed: int,
) -> tf.data.Dataset:
    records = list(total_records) + list(real_train_records)
    paths = [str(record.path) for record in records]
    labels = [record.label for record in records]
    sources = [0 if record.source == "total" else 1 for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, sources))
    dataset = dataset.map(decode_record, num_parallel_calls=4).cache()
    dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)

    total_zoom = tf.keras.layers.RandomZoom(
        height_factor=(-0.1, 0.1),
        width_factor=(-0.1, 0.1),
        fill_mode="constant",
        fill_value=0.0,
        seed=seed,
    )

    def augment_by_source(image: tf.Tensor, label: tf.Tensor, source: tf.Tensor):
        # total 延续原训练的缩放；real 只走受限颜色增强，两者不共享增强层。
        image = tf.cond(
            tf.equal(source, 0),
            lambda: total_zoom(image, training=True),
            lambda: augment_real_image(image),
        )
        return image, label

    dataset = dataset.map(augment_by_source, num_parallel_calls=4)
    return dataset.batch(batch_size).prefetch(4)


def make_labeled_raw_dataset(
    records: Sequence[ImageRecord],
    batch_size: int,
) -> tf.data.Dataset:
    paths = [str(record.path) for record in records]
    labels = [record.label for record in records]
    sources = [1] * len(records)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, sources))
    dataset = dataset.map(decode_record, num_parallel_calls=4).cache()
    dataset = dataset.map(lambda image, label, source: (image, label))
    return dataset.batch(batch_size).prefetch(4)


def make_mmd_dataset(
    records: Sequence[ImageRecord],
    batch_size: int,
    seed: int,
) -> tf.data.Dataset:
    paths = [str(record.path) for record in records]
    labels = [record.label for record in records]
    sources = [1] * len(records)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, sources))
    dataset = dataset.map(decode_record, num_parallel_calls=4).cache()
    dataset = dataset.map(lambda image, label, source: image)
    # MMD 输入只打乱和循环，不执行任何图像增强。
    return (
        dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(4)
    )


def build_mcu_cnn(
    input_shape: tuple[int, int, int] = (120, 120, 3),
    num_classes: int = 10,
) -> tf.keras.Model:
    backbone = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(),
        ],
        name="mcu_cnn_backbone",
    )
    inputs = tf.keras.Input(shape=input_shape)
    backbone_output = backbone(inputs)
    features = tf.keras.layers.Dropout(0.25)(backbone_output)
    logits = tf.keras.layers.Dense(num_classes)(features)
    return tf.keras.Model(inputs=inputs, outputs=[features, logits], name="box_classifier")


def compute_mmd(source_features: tf.Tensor, target_features: tf.Tensor) -> tf.Tensor:
    """Compute mean-heuristic multi-kernel MMD."""

    source_size = tf.shape(source_features)[0]
    target_size = tf.shape(target_features)[0]
    features = tf.concat([source_features, target_features], axis=0)
    dot_products = tf.matmul(features, features, transpose_b=True)
    row_norms = tf.broadcast_to(
        tf.expand_dims(tf.linalg.diag_part(dot_products), 1), tf.shape(dot_products)
    )
    column_norms = tf.broadcast_to(
        tf.expand_dims(tf.linalg.diag_part(dot_products), 0), tf.shape(dot_products)
    )
    distance_squared = tf.maximum(row_norms + column_norms - 2.0 * dot_products, 0.0)
    bandwidth = tf.stop_gradient(tf.reduce_mean(distance_squared))
    bandwidth = tf.maximum(bandwidth, 1e-5)

    kernel = tf.zeros_like(distance_squared)
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        gamma = 1.0 / (2.0 * bandwidth * multiplier)
        kernel += tf.exp(-gamma * distance_squared)

    source_kernel = kernel[:source_size, :source_size]
    target_kernel = kernel[source_size:, source_size:]
    cross_kernel = kernel[:source_size, source_size : source_size + target_size]
    mmd = (
        tf.reduce_mean(source_kernel)
        + tf.reduce_mean(target_kernel)
        - 2.0 * tf.reduce_mean(cross_kernel)
    )
    return tf.maximum(mmd, 0.0)


class MMDDomainAdaptationModel(tf.keras.Model):
    def __init__(self, cnn: tf.keras.Model, mmd_weight: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cnn = cnn
        self.mmd_weight = mmd_weight
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(name="cls_loss")
        self.mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.classification_loss_tracker,
            self.mmd_loss_tracker,
        ]

    def call(self, inputs, training: bool = False):
        return self.cnn(inputs, training=training)

    def train_step(self, data):
        (source_images, source_labels), target_images = data
        with tf.GradientTape() as tape:
            source_features, source_logits = self.cnn(source_images, training=True)
            target_features, _ = self.cnn(target_images, training=True)
            classification_loss = self.compute_loss(
                x=source_images,
                y=source_labels,
                y_pred=source_logits,
                training=True,
            )
            mmd_loss = compute_mmd(source_features, target_features)
            total_loss = classification_loss + self.mmd_weight * mmd_loss

        gradients = tape.gradient(total_loss, self.cnn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.cnn.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.mmd_loss_tracker.update_state(mmd_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "cls_loss": self.classification_loss_tracker.result(),
            "mmd_loss": self.mmd_loss_tracker.result(),
        }

    def test_step(self, data):
        images, labels = data
        _, logits = self.cnn(images, training=False)
        classification_loss = self.compute_loss(
            x=images,
            y=labels,
            y_pred=logits,
            training=False,
        )
        self.total_loss_tracker.update_state(classification_loss)
        return {"loss": self.total_loss_tracker.result()}


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        probabilities = tf.nn.softmax(y_pred, axis=-1)
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.one_hot(labels, depth=tf.shape(y_pred)[-1])
        true_probabilities = tf.reduce_sum(one_hot * probabilities, axis=-1)
        weight = self.alpha * tf.math.pow(1.0 - true_probabilities, self.gamma)
        return K.mean(weight * tf.reshape(cross_entropy, [-1]), axis=-1)


class TopKMacroF1Checkpoints(tf.keras.callbacks.Callback):
    def __init__(self, validation_dataset: tf.data.Dataset, directory: Path, top_k: int):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.directory = directory
        self.top_k = top_k
        self.candidates: list[CandidateCheckpoint] = []

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        y_true: list[int] = []
        y_pred: list[int] = []
        for images, labels in self.validation_dataset:
            _, logits = self.model.cnn(images, training=False)
            y_true.extend(int(value) for value in labels.numpy())
            y_pred.extend(int(value) for value in tf.argmax(logits, axis=-1).numpy())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if logs is not None:
            logs["val_macro_f1"] = macro_f1

        qualifies = len(self.candidates) < self.top_k or macro_f1 > min(
            item.macro_f1 for item in self.candidates
        )
        if qualifies:
            checkpoint_path = self.directory / f"epoch_{epoch + 1:03d}.weights.h5"
            self.model.cnn.save_weights(checkpoint_path)
            self.candidates.append(
                CandidateCheckpoint(epoch=epoch + 1, macro_f1=macro_f1, path=checkpoint_path)
            )
            self.candidates.sort(key=lambda item: (-item.macro_f1, item.epoch))
            while len(self.candidates) > self.top_k:
                removed = self.candidates.pop()
                removed.path.unlink(missing_ok=True)
            print(f"Epoch {epoch + 1:03d}: val_macro_f1={macro_f1:.4f}，保存候选")
        else:
            print(f"Epoch {epoch + 1:03d}: val_macro_f1={macro_f1:.4f}")


def keras_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    labels: list[int] = []
    predictions: list[int] = []
    for images, batch_labels in dataset:
        _, logits = model(images, training=False)
        labels.extend(int(value) for value in batch_labels.numpy())
        predictions.extend(int(value) for value in tf.argmax(logits, axis=-1).numpy())
    return labels, predictions


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, object]:
    recalls = recall_score(
        labels,
        predictions,
        labels=list(range(len(CLASSES))),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class_recall": {
            class_name: float(recalls[index]) for index, class_name in enumerate(CLASSES)
        },
        "min_class_recall": float(np.min(recalls)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(CLASSES)))
        ).tolist(),
        "sample_count": len(labels),
    }


def make_representative_dataset(
    records: Sequence[ImageRecord],
    sample_count: int,
    seed: int,
) -> tf.data.Dataset:
    selected = list(records)
    random.Random(seed).shuffle(selected)
    selected = selected[:sample_count]
    return make_labeled_raw_dataset(selected, batch_size=1)


def export_int8_model(
    model: tf.keras.Model,
    output_path: Path,
    representative_dataset: tf.data.Dataset,
) -> None:
    inputs = tf.keras.Input(shape=(120, 120, 3), name="image")
    _, logits = model(inputs, training=False)
    probabilities = tf.keras.layers.Softmax(name="probabilities")(logits)
    inference_model = tf.keras.Model(inputs=inputs, outputs=probabilities)

    def representative_generator():
        for images, _ in representative_dataset:
            yield [images]

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output_path.write_bytes(converter.convert())


def tflite_predictions(model_path: Path, dataset: tf.data.Dataset):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if input_details["dtype"] != np.int8 or output_details["dtype"] != np.int8:
        raise ValueError("TFLite 模型不是全 INT8 输入输出")

    input_scale, input_zero_point = input_details["quantization"]
    if input_scale <= 0:
        raise ValueError("TFLite 输入量化参数无效")
    labels: list[int] = []
    predictions: list[int] = []
    for images, batch_labels in dataset.unbatch().batch(1):
        quantized = np.rint(images.numpy() / input_scale + input_zero_point)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quantized)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        labels.append(int(batch_labels.numpy()[0]))
        predictions.append(int(np.argmax(output[0])))
    return labels, predictions, {
        "input_dtype": str(input_details["dtype"].__name__),
        "output_dtype": str(output_details["dtype"].__name__),
        "input_shape": [int(value) for value in input_details["shape"]],
        "output_shape": [int(value) for value in output_details["shape"]],
        "input_quantization": [float(input_scale), int(input_zero_point)],
        "output_quantization": [
            float(output_details["quantization"][0]),
            int(output_details["quantization"][1]),
        ],
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def serialize_history(history: dict[str, list[object]]) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in values]
        for key, values in history.items()
    }


def select_int8_candidate(
    candidates: Sequence[CandidateCheckpoint],
    validation_dataset: tf.data.Dataset,
    representative_dataset: tf.data.Dataset,
    candidate_directory: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    for candidate in candidates:
        model = build_mcu_cnn(num_classes=len(CLASSES))
        model.load_weights(candidate.path)
        float_labels, float_predictions = keras_predictions(model, validation_dataset)
        int8_path = candidate_directory / f"epoch_{candidate.epoch:03d}.int8.tflite"
        export_int8_model(model, int8_path, representative_dataset)
        int8_labels, int8_predictions, tensor_details = tflite_predictions(
            int8_path, validation_dataset
        )
        if float_labels != int8_labels:
            raise RuntimeError("float 与 INT8 验证样本顺序不一致")
        result = {
            "epoch": candidate.epoch,
            "checkpoint": str(candidate.path),
            "tflite": str(int8_path),
            "checkpoint_val_macro_f1": candidate.macro_f1,
            "float_validation": classification_metrics(float_labels, float_predictions),
            "int8_validation": classification_metrics(int8_labels, int8_predictions),
            "tflite_tensors": tensor_details,
        }
        results.append(result)

    # 最终候选首先看 INT8 macro-F1，再看最差类别召回率和 accuracy。
    selected = max(
        results,
        key=lambda item: (
            item["int8_validation"]["macro_f1"],
            item["int8_validation"]["min_class_recall"],
            item["int8_validation"]["accuracy"],
            -item["epoch"],
        ),
    )
    return selected, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TOTAL_DIR)
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--real-train-ratio", type=float, default=0.55)
    parser.add_argument("--real-validation-ratio", type=float, default=0.15)
    parser.add_argument("--real-mmd-ratio", type=float, default=0.15)
    parser.add_argument("--real-test-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--representative-samples", type=int, default=500)
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument(
        "--max-total-per-class",
        type=int,
        default=0,
        help="仅用于冒烟测试；0表示使用全部 total 图片",
    )
    parser.add_argument(
        "--max-real-per-class",
        type=int,
        default=0,
        help="仅用于冒烟测试；0表示使用全部 real 图片",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[float, float, float, float]:
    ratios = (
        args.real_train_ratio,
        args.real_validation_ratio,
        args.real_mmd_ratio,
        args.real_test_ratio,
    )
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("batch-size、epochs 和 patience 必须大于0")
    if args.top_k <= 0 or args.representative_samples <= 0:
        raise ValueError("top-k 和 representative-samples 必须大于0")
    if args.mmd_weight < 0:
        raise ValueError("mmd-weight 不能小于0")
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("四项 real 划分比例必须大于0且总和为1")
    return ratios


def prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖已有结果请更换目录: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "candidates").mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()
    ratios = validate_args(args)
    prepare_output_directory(args.output_dir)
    gpu_names = configure_tensorflow(args.seed)

    print("正在扫描并校验数据集...")
    total_records = collect_records(args.data_dir, "total", args.max_total_per_class)
    real_records = collect_records(args.real_dir, "real", args.max_real_per_class)
    real_splits = split_real_records(real_records, ratios, args.seed)
    audit_no_leakage(total_records, real_splits)
    validate_image_shapes(total_records + real_records)
    write_split_manifest(
        args.output_dir / "split_manifest.json",
        total_records,
        real_splits,
        ratios,
        args.seed,
    )

    run_config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "classes": list(CLASSES),
        "image_size": list(IMAGE_SIZE),
        "python": sys.executable,
        "tensorflow": tf.__version__,
        "gpus": gpu_names,
        "real_augmentation": REAL_AUGMENTATION_SPEC,
        "real_counts": {
            name: count_by_class(records) for name, records in real_splits.items()
        },
        "total_counts": count_by_class(total_records),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps(run_config["real_counts"], indent=2, ensure_ascii=False))

    source_dataset = make_source_dataset(
        total_records, real_splits["train"], args.batch_size, args.seed
    )
    validation_dataset = make_labeled_raw_dataset(
        real_splits["validation"], args.batch_size
    )
    mmd_dataset = make_mmd_dataset(real_splits["mmd"], args.batch_size, args.seed)
    test_dataset = make_labeled_raw_dataset(real_splits["test"], args.batch_size)
    training_dataset = tf.data.Dataset.zip((source_dataset, mmd_dataset)).prefetch(4)

    base_cnn = build_mcu_cnn(num_classes=len(CLASSES))
    model = MMDDomainAdaptationModel(base_cnn, mmd_weight=args.mmd_weight)
    model(tf.zeros((1, 120, 120, 3), dtype=tf.float32))
    base_cnn.summary()

    steps_per_epoch = int(tf.data.experimental.cardinality(source_dataset).numpy())
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=steps_per_epoch * args.epochs,
        alpha=0.01,
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
        ),
        loss=SparseCategoricalFocalLoss(gamma=2.0, alpha=1.0),
    )

    checkpoint_callback = TopKMacroF1Checkpoints(
        validation_dataset=validation_dataset,
        directory=args.output_dir / "candidates",
        top_k=min(args.top_k, args.epochs),
    )
    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        shuffle=False,
        callbacks=[
            checkpoint_callback,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_macro_f1",
                mode="max",
                patience=args.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2,
    )
    write_json(args.output_dir / "history.json", serialize_history(history.history))
    if not checkpoint_callback.candidates:
        raise RuntimeError("训练没有生成候选权重")

    representative_dataset = make_representative_dataset(
        real_splits["train"], args.representative_samples, args.seed
    )
    selected, candidate_results = select_int8_candidate(
        checkpoint_callback.candidates,
        validation_dataset,
        representative_dataset,
        args.output_dir / "candidates",
    )
    selected_weights = Path(selected["checkpoint"])
    selected_tflite = Path(selected["tflite"])
    final_weights = args.output_dir / "best.weights.h5"
    final_tflite = args.output_dir / "best_int8.tflite"
    shutil.copy2(selected_weights, final_weights)
    shutil.copy2(selected_tflite, final_tflite)

    final_model = build_mcu_cnn(num_classes=len(CLASSES))
    final_model.load_weights(final_weights)
    float_labels, float_predictions = keras_predictions(final_model, test_dataset)
    int8_labels, int8_predictions, tensor_details = tflite_predictions(
        final_tflite, test_dataset
    )
    if float_labels != int8_labels:
        raise RuntimeError("float 与 INT8 测试样本顺序不一致")
    float_metrics = classification_metrics(float_labels, float_predictions)
    int8_metrics = classification_metrics(int8_labels, int8_predictions)

    metrics = {
        "selection_rule": ["int8_macro_f1", "int8_min_class_recall", "int8_accuracy"],
        "selected_epoch": selected["epoch"],
        "selected_validation": selected,
        "candidates": candidate_results,
        "test": {
            "float": float_metrics,
            "int8": int8_metrics,
            "int8_minus_float_accuracy": int8_metrics["accuracy"]
            - float_metrics["accuracy"],
            "float_int8_prediction_agreement": float(
                np.mean(np.asarray(float_predictions) == np.asarray(int8_predictions))
            ),
        },
        "tflite": {
            **tensor_details,
            "path": str(final_tflite),
            "size_bytes": final_tflite.stat().st_size,
        },
    }
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics["test"], indent=2, ensure_ascii=False))
    print(f"最终模型: {final_tflite}")


if __name__ == "__main__":
    main()
