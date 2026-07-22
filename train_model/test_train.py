from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import (
    CLASSES,
    REAL_AUGMENTATION_SPEC,
    ImageRecord,
    audit_no_leakage,
    make_mmd_dataset,
    normalize_capture_stem,
    split_real_records,
)


class RealSplitTest(unittest.TestCase):
    def make_records(self, root: Path) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        for label, class_name in enumerate(CLASSES):
            for index in range(20):
                path = root / class_name / f"{index}.jpg"
                records.append(
                    ImageRecord(
                        path=path,
                        class_name=class_name,
                        label=label,
                        source="real",
                        sha256=f"{class_name}-{index}",
                    )
                )
        return records

    def test_filename_variants_have_same_normalized_stem(self) -> None:
        self.assertEqual(
            normalize_capture_stem(Path("35.jpg")),
            normalize_capture_stem(Path("35 (2).jpg")),
        )

    def test_split_is_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            records = self.make_records(Path(temporary_directory))
            first = split_real_records(records, (0.55, 0.15, 0.15, 0.15), seed=123)
            second = split_real_records(records, (0.55, 0.15, 0.15, 0.15), seed=123)
            self.assertEqual(first, second)
            audit_no_leakage([], first)
            for split_records in first.values():
                self.assertEqual({record.class_name for record in split_records}, set(CLASSES))

    def test_training_hash_cannot_leak_into_held_out_real(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            records = self.make_records(Path(temporary_directory))
            splits = split_real_records(records, (0.55, 0.15, 0.15, 0.15), seed=123)
            leaked = splits["validation"][0]
            total_record = ImageRecord(
                path=Path(temporary_directory) / "total.jpg",
                class_name=leaked.class_name,
                label=leaked.label,
                source="total",
                sha256=leaked.sha256,
            )
            with self.assertRaisesRegex(ValueError, "训练数据"):
                audit_no_leakage([total_record], splits)


class RealAugmentationTest(unittest.TestCase):
    def test_real_augmentation_stays_within_shotmix_foreground_limits(self) -> None:
        self.assertLessEqual(REAL_AUGMENTATION_SPEC["brightness_limit"], 0.08)
        self.assertLessEqual(REAL_AUGMENTATION_SPEC["contrast_limit"], 0.08)
        self.assertLessEqual(REAL_AUGMENTATION_SPEC["hue_shift_limit"], 2.0)
        self.assertLessEqual(REAL_AUGMENTATION_SPEC["saturation_shift_limit"], 4.0)
        self.assertLessEqual(REAL_AUGMENTATION_SPEC["value_shift_limit"], 3.0)
        prohibited = {"blur", "compression", "downscale", "noise", "cutout", "zoom"}
        self.assertTrue(prohibited.isdisjoint(REAL_AUGMENTATION_SPEC))

    def test_mmd_dataset_keeps_decoded_pixels_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.png"
            original = np.arange(120 * 120 * 3, dtype=np.uint8).reshape((120, 120, 3))
            tf.io.write_file(str(path), tf.io.encode_png(original))
            record = ImageRecord(
                path=path,
                class_name="00",
                label=0,
                source="real",
                sha256="sample",
                group_id="00:sample",
            )
            batch = next(iter(make_mmd_dataset([record], batch_size=1, seed=123)))
            np.testing.assert_array_equal(batch.numpy()[0], original.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
