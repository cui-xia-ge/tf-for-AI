"""Mix OpenART-captured foregrounds with augmented real camera backgrounds."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


DEFAULT_BG_DIR = Path(r"D:\college\718\dataset\box\background")
DEFAULT_FG_DIR = Path(r"D:\college\718\dataset\box\shot_front")
#DEFAULT_FG_DIR = Path(r"D:\college\718\智能视觉调试环境搭建软件\上位机\调试版本\SmartCar_VR_V1.6\SmartCar_VR_V1.6\image_class")
DEFAULT_OUT_DIR = Path(r"D:\college\718\dataset\box\mix")

IMAGE_WIDTH = 120
IMAGE_HEIGHT = 120
MIN_FG_SIZE = 50
MAX_FG_SIZE = 115
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES = tuple((str(index), f"{index:02d}") for index in range(10))


def apply_directional_illumination(image: np.ndarray, **kwargs) -> np.ndarray:
    """Apply a smooth light gradient and mild lens shading to a real background."""

    height, width = image.shape[:2]
    yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    angle = random.uniform(0.0, 2.0 * np.pi)
    strength = random.uniform(0.06, 0.18)
    directional = 1.0 + strength * (np.cos(angle) * xx + np.sin(angle) * yy)

    center_x = random.uniform(-0.20, 0.20)
    center_y = random.uniform(-0.20, 0.20)
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    vignette_strength = random.uniform(0.02, 0.12)
    vignette = 1.0 - vignette_strength * np.clip(radius / np.sqrt(2.0), 0.0, 1.0) ** 2

    gain = directional * vignette
    gain /= max(float(gain.mean()), 1e-6)
    result = image.astype(np.float32) * gain[..., np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def rgb565_roundtrip(image: np.ndarray, **kwargs) -> np.ndarray:
    """Simulate the quantization used by both OpenART and SCC8660 RGB565 frames."""

    result = image.copy()
    red5 = image[..., 0] >> 3
    green6 = image[..., 1] >> 2
    blue5 = image[..., 2] >> 3
    result[..., 0] = (red5 << 3) | (red5 >> 2)
    result[..., 1] = (green6 << 2) | (green6 >> 4)
    result[..., 2] = (blue5 << 3) | (blue5 >> 2)
    return result


# Background-only augmentation. It deliberately keeps the 120x120 geometry and
# changes only factors that vary between real captures.
BACKGROUND_PIPELINE = A.Compose(
    [
        A.HorizontalFlip(p=0.35),
        A.Lambda(image=apply_directional_illumination, p=0.35),
        A.RandomBrightnessContrast(
            brightness_limit=0.12, contrast_limit=0.10, p=0.65
        ),
        A.OneOf(
            [
                A.RGBShift(
                    r_shift_limit=(-6, 8),
                    g_shift_limit=(-5, 6),
                    b_shift_limit=(-8, 8),
                    p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=3,
                    sat_shift_limit=8,
                    val_shift_limit=5,
                    p=1.0,
                ),
            ],
            p=0.40,
        ),
        A.OneOf(
            [
                A.RandomGamma(gamma_limit=(88, 112), p=1.0),
                A.CLAHE(clip_limit=(1.0, 1.8), tile_grid_size=(4, 4), p=1.0),
            ],
            p=0.15,
        ),
    ]
)


# Foregrounds are already OpenART captures, so their augmentation is kept mild.
FOREGROUND_PIPELINE = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=0.08, contrast_limit=0.08, p=0.30
        ),
        A.OneOf(
            [
                A.RGBShift(
                    r_shift_limit=(-3, 4),
                    g_shift_limit=(-3, 3),
                    b_shift_limit=(-4, 4),
                    p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=2,
                    sat_shift_limit=4,
                    val_shift_limit=3,
                    p=1.0,
                ),
            ],
            p=0.20,
        ),
        A.GaussianBlur(blur_limit=(3, 3), p=0.10),
    ]
)


# Capture-level augmentation is applied after compositing so foreground and
# background receive the same sensor artifacts and the pasted edge is softened.
CAPTURE_PIPELINE = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=0.04, contrast_limit=0.04, p=0.20
        ),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ],
            p=0.18,
        ),

        A.OneOf(
            [
                A.GaussNoise(std_range=(0.01, 0.05),  per_channel=False, noise_scale_factor=0.6, p=1.0),
                A.ISONoise(
                    color_shift=(0.01, 0.05), intensity=(0.05, 0.10), p=1.0
                ),
            ],
            p=0.15
        ),
        A.OneOf([
            A.Downscale(scale_range=(0.7, 0.9), interpolation_pair={"upscale":cv2.INTER_NEAREST,"downscale":cv2.INTER_LINEAR},p=1),
            A.Downscale(scale_range=(0.7, 0.9), interpolation_pair={"upscale":cv2.INTER_LINEAR,"downscale":cv2.INTER_NEAREST},p=1),
        ],p=0.25),
        A.Lambda(image=rgb565_roundtrip, 
                 p=0.35
        ),
        A.ImageCompression(quality_lower=65, quality_upper=95, 
                           p=0.40),
    ]
)


def get_image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(
    path: Path, mode: str, expected_size: tuple[int, int] | None = None
) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert(mode)
    if expected_size is not None and image.size != expected_size:
        raise ValueError(
            f"图片尺寸必须为 {expected_size[0]}x{expected_size[1]}: "
            f"{path} -> {image.size}"
        )
    return image


def apply_rgba_pipeline(image: Image.Image) -> Image.Image:
    rgba = np.asarray(image, dtype=np.uint8)
    augmented_rgb = FOREGROUND_PIPELINE(image=rgba[..., :3])["image"]
    return Image.fromarray(np.dstack((augmented_rgb, rgba[..., 3])), mode="RGBA")


def feather_alpha_edges(image: Image.Image, radius: int) -> Image.Image:
    """Fade JPEG foreground borders to avoid a perfectly sharp pasted rectangle."""

    if radius <= 0:
        return image
    rgba = np.asarray(image, dtype=np.uint8).copy()
    height, width = rgba.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    edge_distance = np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy))
    ramp = np.clip(edge_distance.astype(np.float32) / radius, 0.0, 1.0)
    rgba[..., 3] = np.rint(rgba[..., 3].astype(np.float32) * ramp).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def cutout_foreground(
    image: Image.Image,
    probability: float = 0.12,
    scale: tuple[float, float] = (0.02, 0.06),
) -> Image.Image:
    """Reveal the augmented background through one small foreground occlusion."""

    result = image.copy()
    if random.random() > probability:
        return result
    width, height = result.size
    target_area = random.uniform(*scale) * width * height
    aspect_ratio = random.uniform(0.35, 2.85)
    cut_height = max(1, min(height - 1, round(np.sqrt(target_area * aspect_ratio))))
    cut_width = max(1, min(width - 1, round(np.sqrt(target_area / aspect_ratio))))
    left = random.randint(0, width - cut_width)
    top = random.randint(0, height - cut_height)
    draw = ImageDraw.Draw(result)
    draw.rectangle(
        [left, top, left + cut_width - 1, top + cut_height - 1],
        fill=(0, 0, 0, 0),
    )
    return result


def augment_background(image: Image.Image, enabled: bool) -> Image.Image:
    if not enabled:
        return image.copy()
    result = BACKGROUND_PIPELINE(image=np.asarray(image, dtype=np.uint8))["image"]
    return Image.fromarray(result, mode="RGB")


def augment_capture(image: Image.Image, enabled: bool) -> Image.Image:
    if not enabled:
        return image.copy()
    result = CAPTURE_PIPELINE(image=np.asarray(image, dtype=np.uint8))["image"]
    if result.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise RuntimeError(f"整图增强改变了输出尺寸: {result.shape}")
    return Image.fromarray(result, mode="RGB")


def load_backgrounds(paths: list[Path]) -> list[Image.Image]:
    """Validate and cache the small 120x120 background set in memory."""

    if not paths:
        raise RuntimeError("实拍背景目录中没有找到图片")
    return [
        load_image(path, "RGB", expected_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        for path in paths
    ]


def generate_dataset(args: argparse.Namespace) -> None:
    if not 1 <= args.min_fg_size <= args.max_fg_size <= IMAGE_WIDTH:
        raise ValueError("前景尺寸必须满足 1 <= min <= max <= 120")
    if args.samples_per_foreground <= 0:
        raise ValueError("--samples-per-foreground 必须大于0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    background_paths = get_image_paths(args.background_dir)
    backgrounds = load_backgrounds(background_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    for folder_name, class_id in CLASSES:
        foreground_dir = args.foreground_dir / folder_name
        foreground_paths = get_image_paths(foreground_dir)
        if args.max_per_class > 0:
            foreground_paths = foreground_paths[: args.max_per_class]
        if not foreground_paths:
            print(f"警告: {foreground_dir} 没有图片，跳过类别 {class_id}")
            continue
        random.shuffle(foreground_paths)
        output_dir = args.output_dir / class_id
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = 0

        for foreground_path in foreground_paths:
            foreground_original = load_image(foreground_path, "RGBA")
            for _ in range(args.samples_per_foreground):
                background = augment_background(
                    random.choice(backgrounds),
                    enabled=not args.no_background_augmentation,
                )

                target_size = random.randint(args.min_fg_size, args.max_fg_size)
                scale = target_size / max(foreground_original.size)
                foreground_size = (
                    max(1, round(foreground_original.width * scale)),
                    max(1, round(foreground_original.height * scale)),
                )
                foreground = foreground_original.resize(
                    foreground_size, Image.Resampling.LANCZOS
                )
                foreground = apply_rgba_pipeline(foreground)
                foreground = feather_alpha_edges(
                    foreground, radius=random.randint(2, 5)
                )
                foreground = cutout_foreground(foreground)

                max_x = IMAGE_WIDTH - foreground.width
                max_y = IMAGE_HEIGHT - foreground.height
                position = (
                    random.randint(0, max_x),
                    random.randint(0, max_y),
                )
                background.paste(foreground, position, mask=foreground.getchannel("A"))
                output = augment_capture(
                    background, enabled=not args.no_capture_augmentation
                )

                # Preserve the original behavior: reruns overwrite the same
                # sequential filenames instead of creating unique variants.
                output_path = output_dir / f"shotmix_{class_id}_{generated:04d}.jpg"
                output.save(output_path, quality=95, subsampling=0, optimize=True)
                generated += 1
                total_generated += 1

        print(
            f"类别 {class_id} 完成: {generated} 张，"
            f"同名文件已直接覆盖 -> {output_dir}"
        )

    print(f"全部完成，共生成 {total_generated} 张120x120图片")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BG_DIR)
    parser.add_argument("--foreground-dir", type=Path, default=DEFAULT_FG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-fg-size", type=int, default=MIN_FG_SIZE)
    parser.add_argument("--max-fg-size", type=int, default=MAX_FG_SIZE)
    parser.add_argument("--samples-per-foreground", type=int, default=1)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="仅用于预览/测试；0表示使用该类别全部图片",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-background-augmentation", action="store_true")
    parser.add_argument("--no-capture-augmentation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate_dataset(parse_args())
