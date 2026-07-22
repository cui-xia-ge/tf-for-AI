"""Fill each real-capture class with deterministic ShotMix samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is only needed for deep validation.
    Image = None


CLASSES = tuple(f"{index:02d}" for index in range(10))
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
SHOTMIX_PREFIX = "shotmix_"
DEFAULT_REAL_DIR = Path("/home/cgcgs/718/dataset/box/real")
DEFAULT_TOTAL_DIR = Path("/home/cgcgs/718/dataset/box/total")
DEFAULT_MANIFEST_NAME = ".openart_fill_manifest.json"


def image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def class_paths(root: Path, class_name: str) -> list[Path]:
    class_dir = root / class_name
    if not class_dir.is_dir():
        raise ValueError(f"missing class directory: {class_dir}")
    return image_paths(class_dir)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> None:
    if path.stat().st_size <= 0:
        raise ValueError(f"empty image: {path}")
    if Image is None:
        return
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise ValueError(f"invalid image: {path}") from error


def load_existing_manifest(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid manifest JSON: {path}") from error
    if not isinstance(value, dict) or not isinstance(value.get("target_count"), int):
        raise ValueError(f"manifest has no integer target_count: {path}")
    if value.get("classes") != list(CLASSES):
        raise ValueError(f"manifest classes do not match {CLASSES}: {path}")
    return value


def calculate_target_count(
    real_dir: Path, total_dir: Path
) -> tuple[int, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, int]] = {}
    totals: list[int] = []
    for class_name in CLASSES:
        real = class_paths(real_dir, class_name)
        shotmix = [
            path
            for path in class_paths(total_dir, class_name)
            if path.name.startswith(SHOTMIX_PREFIX)
        ]
        # 目标数量必须基于“原始实拍 + total 中全部 ShotMix”的可用总量。
        # 复制后的 ShotMix 已经出现在 real_dir 中，不能在重跑时重复计数。
        original_real_count = sum(
            1 for path in real if not path.name.startswith(SHOTMIX_PREFIX)
        )
        counts[class_name] = {
            "real_count": len(real),
            "original_real_count": original_real_count,
            "shotmix_count": len(shotmix),
            "available_total": original_real_count + len(shotmix),
        }
        totals.append(counts[class_name]["available_total"])
    return min(totals), counts


def prepare_dataset(
    real_dir: Path,
    total_dir: Path,
    manifest_path: Path,
    seed: int,
    dry_run: bool,
) -> dict[str, object]:
    existing = load_existing_manifest(manifest_path)
    if existing is None:
        target_count, counts = calculate_target_count(real_dir, total_dir)
    else:
        target_count = int(existing["target_count"])
        _, counts = calculate_target_count(real_dir, total_dir)

    if target_count <= 0:
        raise ValueError("calculated target_count must be positive")

    copied: list[dict[str, str]] = []
    planned_counts: dict[str, int] = {}
    for class_index, class_name in enumerate(CLASSES):
        destination_dir = real_dir / class_name
        destination_paths = class_paths(real_dir, class_name)
        if len(destination_paths) > target_count:
            raise ValueError(
                f"class {class_name} already has {len(destination_paths)} images, "
                f"above target {target_count}"
            )
        missing = target_count - len(destination_paths)
        planned_counts[class_name] = missing
        if missing == 0:
            continue

        existing_by_name = {path.name: path for path in destination_paths}
        candidates = [
            path
            for path in class_paths(total_dir, class_name)
            if path.name.startswith(SHOTMIX_PREFIX)
        ]
        # 每类使用独立但可复现的随机序列，避免类别目录排序影响选择结果。
        random.Random(seed + class_index).shuffle(candidates)
        selected: list[Path] = []
        for source in candidates:
            destination = destination_dir / source.name
            if source.name in existing_by_name:
                if sha256(source) != sha256(destination):
                    raise ValueError(
                        f"destination collision differs from source: {destination}"
                    )
                continue
            selected.append(source)
            if len(selected) == missing:
                break
        if len(selected) < missing:
            raise ValueError(
                f"class {class_name} needs {missing} ShotMix images, "
                f"but only {len(selected)} unused candidates are available"
            )

        for source in selected:
            # 复制前校验源文件；Pillow 不存在时仍会拒绝空文件。
            validate_image(source)
            destination = destination_dir / source.name
            copied.append(
                {
                    "class": class_name,
                    "source": str(source),
                    "destination": str(destination),
                    "sha256": sha256(source),
                }
            )
            if not dry_run:
                shutil.copy2(source, destination)

    result: dict[str, object] = {
        "real_dir": str(real_dir),
        "total_dir": str(total_dir),
        "manifest": str(manifest_path),
        "classes": list(CLASSES),
        "target_count": target_count,
        "seed": seed,
        "dry_run": dry_run,
        "source_counts": counts,
        "planned_missing_counts": planned_counts,
        "copied": copied,
    }
    if not dry_run:
        for class_name in CLASSES:
            final_count = len(class_paths(real_dir, class_name))
            if final_count != target_count:
                raise RuntimeError(
                    f"class {class_name} ended with {final_count}, "
                    f"expected {target_count}"
                )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--total-dir", type=Path, default=DEFAULT_TOTAL_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest or args.real_dir / DEFAULT_MANIFEST_NAME
    result = prepare_dataset(
        real_dir=args.real_dir,
        total_dir=args.total_dir,
        manifest_path=manifest,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
