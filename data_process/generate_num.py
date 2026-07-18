import os
import random
import glob
import numpy as np
import cv2
import albumentations as A
from PIL import Image, ImageDraw, ImageFont

# ================= 核心配置区 =================
OUTPUT_DIR = "D:/college/718/dataset/digit_dataset"  # 输出文件夹名
IMAGE_SIZE = 28  # 图像分辨率
SAMPLES_PER_DIGIT = 160  # 每个数字生成的数量
FONTS_DIR = "D:/college/718/dataset/fonts"  # 存放 .ttf / .ttc 字体文件的目录
# ==============================================

def setup_directories():
    """创建 0-8 的子文件夹"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # 只生成 0 到 8
    for i in range(9):
        folder_path = os.path.join(OUTPUT_DIR, str(i))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def get_font_list():
    """获取字体文件"""
    fonts = glob.glob(os.path.join(FONTS_DIR, "*.ttf")) + glob.glob(os.path.join(FONTS_DIR, "*.ttc"))
    if not fonts:
        print(f"❌ 错误: 在 '{FONTS_DIR}' 文件夹中没有找到任何字体文件！")
        exit(1)
    return fonts

def get_albumentations_pipeline():
    """配置 Albumentations 数据增强流水线"""
    return A.Compose([
        # p=1.0 保证每次都会进入这个随机选择池
        A.RandomRotate90(p=1.0)
    ])


def generate_digit_image(digit_str, font_path, save_path, transform):
    """直接以灰度模式生成并增强单张图片"""

    # 👇 核心修改点：缩小基础字体比例，留出安全边距
    font_size = random.randint(int(IMAGE_SIZE * 0.7), int(IMAGE_SIZE * 0.9))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"无法加载字体 {font_path}: {e}")
        return

    # 1. 直接创建灰度模式 'L' 的 PIL 图像 (背景纯黑 0)
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=0)
    draw = ImageDraw.Draw(img)

    # 2. 居中绘制纯白文字 (255)
    try:
        bbox = draw.textbbox((0, 0), digit_str, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        base_x = (IMAGE_SIZE - text_w) / 2 - bbox[0]
        base_y = (IMAGE_SIZE - text_h) / 2 - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(digit_str, font=font)
        base_x = (IMAGE_SIZE - text_w) / 2
        base_y = (IMAGE_SIZE - text_h) / 2

    draw.text((base_x, base_y), digit_str, font=font, fill=255)

    # 3. PIL 格式转为 Numpy 数组
    image_np = np.array(img)

    # 4. 执行数据增强
    augmented = transform(image=image_np)
    aug_image_np = augmented['image']

    # 5. 直接转回 PIL 并保存
    final_img = Image.fromarray(aug_image_np)
    final_img.save(save_path)

def main():
    setup_directories()
    font_list = get_font_list()
    transform = get_albumentations_pipeline()

    print(f"✅ 找到 {len(font_list)} 种字体，开始生成高鲁棒性数据集...")

    total_generated = 0
    # 遍历 0 到 8
    for digit in range(9):
        digit_str = str(digit)
        for i in range(SAMPLES_PER_DIGIT):
            font_path = random.choice(font_list)
            filename = f"{digit}_{i:04d}.jpg"
            save_path = os.path.join(OUTPUT_DIR, digit_str, filename)

            generate_digit_image(digit_str, font_path, save_path, transform)
            total_generated += 1

        print(f"👉 数字 {digit} 生成完毕 ({SAMPLES_PER_DIGIT} 张)")

    print("-" * 30)
    print(f"🎉 搞定！总共生成了 {total_generated} 张 {IMAGE_SIZE}x{IMAGE_SIZE} 的灰度图。")


if __name__ == "__main__":
    main()