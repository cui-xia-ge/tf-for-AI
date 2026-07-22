import os
import random
import glob
import random
import cv2
from PIL import Image, ImageDraw
import numpy as np
import albumentations as A

# ================= 配置区域 =================
# 文件夹路径
BG_DIR = "D:\\college\\718\\dataset\\box\\background"  # 背景图片文件夹路径
BASE_FG_DIR = "D:\\college\\718\\seekfree\\官方数据集"  # 前景图片根路径
BASE_OUT_DIR = "D:\\college\\718\\dataset\\box\\mix"  # 合成数据集保存根路径

# 尺寸设置
MAX_FG_SIZE = 115
MIN_FG_SIZE = 50
# 背景尺寸设定
BG_WIDTH = 120
BG_HEIGHT = 120
CLASSES = [
    ("00mickey_mouse", "00"),
    ("01pikachu", "01"),
    ("02spongebob_squarepants", "02"),
    ("03pleasant_sheep", "03"),
    ("04donald_duck", "04"),
    ("05nezha", "05"),
    ("06big_head_son", "06"),
    ("07gg_bond", "07"),
    ("08calabash_brothers", "08"),
    ("09grey_wolf", "09")
]
# ============================================

def cutout_fg_pil(fg_img, p=0.5, scale=(0.05, 0.2)):
    """
    专门针对 PIL RGBA 图像的 Cutout
    p: 触发概率
    scale: 遮挡面积占前景面积的比例范围
    """
    if random.random() > p:
        return fg_img
    w, h = fg_img.size
    area = w * h

    # 随机生成遮挡块的面积和长宽比
    target_area = random.uniform(*scale) * area
    aspect_ratio = random.uniform(0.1, 10.0)

    cut_h = int(np.sqrt(target_area * aspect_ratio))
    cut_w = int(np.sqrt(target_area / aspect_ratio))
    # 边界检查
    cut_h = min(cut_h, h - 5)
    cut_w = min(cut_w, w - 5)
    # 随机位置
    x = random.randint(0, w - cut_w)
    y = random.randint(0, h - cut_h)

    # 在 Alpha 通道上画一个透明矩形 (R, G, B, Alpha=0)
    draw = ImageDraw.Draw(fg_img)
    draw.rectangle([x, y, x + cut_w, y + cut_h], fill=(0, 0, 0, 0))
    return fg_img

def get_image_paths(directory):
    extensions = '*.jpg'
    paths = []
    paths.extend(glob.glob(os.path.join(directory, extensions)))
    return paths


def apply_screen_grid(image, **kwargs):
    """
    模拟物理屏幕的 RGB 子像素“纱窗效应”(Screen Door Effect)
    """
    img_np = np.array(image, dtype=np.float32)
    h, w = img_np.shape[:2]

    # 随机生成网格的间距 (模拟不同距离拍摄屏幕)
    grid_size = random.randint(2, 4)

    # 创建一个与原图同大小的全 1 遮罩
    mask = np.ones((h, w, 3), dtype=np.float32)

    # 每隔 grid_size 个像素，画一条微弱的黑线 (横竖交叉形成网格)
    darkness = random.uniform(0.7, 0.9)  # 网格的黑度，越小越黑
    mask[::grid_size, :, :] = darkness
    mask[:, ::grid_size, :] = darkness

    # 将网格遮罩乘到原图上
    gridded_img = (img_np * mask).astype(np.uint8)
    return gridded_img
# 自定义暗角生成器
def apply_screen_vignette(image, **kwargs):
    """
    高级屏幕暗角模拟器 (V2.0)
    引入了“中心亮区保护”和“归一化平滑衰减”逻辑
    """
    h, w = image.shape[:2]

    # 1. 生成坐标网格 (-1 到 1)
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    distance = np.sqrt(X ** 2 + Y ** 2)

    # --- 随机参数生成 ---
    intensity = random.uniform(0.2, 0.6)  # 边缘最大变暗程度
    falloff = random.uniform(1, 2.5)  # 衰减的曲线陡峭度
    center_radius = random.uniform(0.3, 0.6)  # 新增：中心 100% 高亮的受保护半径

    # 2. 初始化全亮的遮罩
    mask = np.ones((h, w), dtype=np.float32)

    # 3. 找出需要变暗的区域 (即距离大于 center_radius 的区域)
    decay_zone = distance > center_radius

    # 4. 在衰减区内，重新计算归一化的距离 (从 0 开始算起)
    # 矩形对角线最大距离约为 1.414，所以最大跨度是 (1.414 - center_radius)
    max_distance = 1.414
    normalized_dist = (distance[decay_zone] - center_radius) / (max_distance - center_radius)

    # 5. 应用衰减曲线 (模拟前端的平滑过渡)
    # 只有处于 decay_zone 的像素才会计算变暗
    decay_factor = (normalized_dist ** falloff) * intensity
    mask[decay_zone] = 1.0 - decay_factor

    # 防止过度死黑
    mask = np.clip(mask, 0.15, 1.0)

    # 6. 应用遮罩
    darkened_image = (image * mask[..., np.newaxis]).astype(np.uint8)

    return darkened_image

# --- 管道 1：前景基础增强（保持物体本身的多样性） ---
fg_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1,0.1), p=1),
    # 模拟屏幕像素点和轻微摩尔纹 (先缩小再放大，产生由于拍摄屏幕导致的颗粒感)
    #interpolation_pair={"upscale":cv2.INTER_LINEAR,"downscale":cv2.INTER_LINEAR}
    A.Downscale(scale_range=(0.7, 0.9), p=0.8),

    A.Lambda(image=apply_screen_grid, p=0.5),
    # 专门针对白平衡漂移的增强模块
    A.OneOf([
        # 1. 偏冷调 (模拟摄像头遇到暖屏，强行加蓝)
        # 压低红绿通道，抬高蓝通道
        A.RGBShift(r_shift_limit=(-2, 0), g_shift_limit=(-2, 0), b_shift_limit=(0, 5), p=1.0),
        # 2. 偏暖调 (模拟摄像头遇到冷屏，强行加黄/红)
        # 抬高红绿通道，压低蓝通道
        A.RGBShift(r_shift_limit=(0, 2), g_shift_limit=(0, 2), b_shift_limit=(-5, 0), p=1.0),
        # 3. 偏绿调 (非常经典的单片机 OV 摄像头偏色)
        # 抬高绿通道
        A.RGBShift(r_shift_limit=(-2, 2), g_shift_limit=(0, 5), b_shift_limit=(-2, 2), p=1.0),
        # 4. 全局随机轻微色相抖动 (作为补充)
        A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=2, val_shift_limit=0, p=1.0)
    ], p=0.3),
    A.GaussianBlur(blur_limit=(3,3), p=0.4),
    A.OneOf([
        A.Defocus(radius=3, alias_blur=(0.1, 0.2), p=0.4),
        # 变焦模糊/径向模糊
        A.ZoomBlur(max_factor=1.05, p=0.4),
    ], p=0.5),
    A.ISONoise(color_shift=(0.05, 0.15), intensity=(0.2, 0.5), p=0.5),  # ISO 噪声
    #JPEG 压缩伪影 (模拟廉价摄像头 ISP 处理后产生的块状模糊)
    #quality_range=(15,30)比较像！！！
    A.ImageCompression(quality_range=(15,50), p=0.7),
])

def apply_fg_albumentations(fg_pil):
    fg_np = np.array(fg_pil)
    if len(fg_np.shape) == 3 and fg_np.shape[2] == 4:
        rgb = fg_np[:, :, :3]
        alpha = fg_np[:, :, 3]
        augmented = fg_pipeline(image=rgb)["image"]
        fg_np = np.dstack((augmented, alpha))
    else:
        fg_np = fg_pipeline(image=fg_np)["image"]
    return Image.fromarray(fg_np)

# --- 管道 2：[新增] 屏幕翻拍全局增强（作用于整张合成图） ---
screen_pipeline = A.Compose([
    # 模拟摄像头不垂直于屏幕造成的透视畸变 (强烈推荐)
    A.Perspective(scale=(0, 0.05), p=1),
    A.Lambda(image=apply_screen_vignette, p=1),
])

def apply_screen_albumentations(bg_pil):
    bg_np = np.array(bg_pil)
    augmented = screen_pipeline(image=bg_np)["image"]
    return Image.fromarray(augmented)

def generate_dataset():
    bg_paths = get_image_paths(BG_DIR)
    if not bg_paths:
        print("错误：背景文件夹中没有找到图片！")
        return
    for folder_name, class_id in CLASSES:
        fg_dir = os.path.join(BASE_FG_DIR, folder_name)
        out_dir = os.path.join(BASE_OUT_DIR, class_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        fg_paths = get_image_paths(fg_dir)
        if not fg_paths:
            print(f"警告：前景文件夹 {fg_dir} 中没有找到图片，跳过此类别！")
            continue

        total_fg = len(fg_paths)
        print(f"开始生成类别 [{class_id} - {folder_name}] 的数据集，预计生成 {total_fg} 张...")
        random.shuffle(fg_paths)
        # 记录当前生成的总图片数，用于命名
        generate_count = 0

        for fg_img_path in fg_paths:
            # 🚀 优化：在内层循环外读取前景原图，避免重复读取硬盘，极大提高生成速度
            fg_original = Image.open(fg_img_path)
            if fg_original.mode != 'RGBA':
                fg_original = fg_original.convert("RGBA")

            bg_img_path = random.choice(bg_paths)
            bg = Image.open(bg_img_path).convert("RGB")

            if bg.size != (BG_WIDTH, BG_HEIGHT):
                bg = bg.resize((BG_WIDTH, BG_HEIGHT), Image.Resampling.LANCZOS)

            # 1. 随机缩放前景 (基于干净的原图 fg_original 每次重新生成)
            new_size = random.randint(MIN_FG_SIZE, MAX_FG_SIZE)
            fg = fg_original.resize((new_size, new_size), Image.Resampling.LANCZOS)

            # 2. 前景专属数据增强
            fg = apply_fg_albumentations(fg)

            # 3. 引入随机旋转
            #angle = random.uniform(-5.0, 5.0)
            #fg = fg.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

            # 设定 50% 概率出现遮挡，遮挡面积为前景的 5%~20%
            fg = cutout_fg_pil(fg, p=0.5, scale=(0.05, 0.2))
            # 4. 将前景贴到背景上
            fg_w, fg_h = fg.size
            max_x = BG_WIDTH - fg_w
            max_y = BG_HEIGHT - fg_h
            pos_x = random.randint(0, max(0, max_x))
            pos_y = random.randint(0, max(0, max_y))
            bg.paste(fg, (pos_x, pos_y), mask=fg)

            # 5. 灵魂操作：对贴好之后的整张图，施加“屏幕翻拍滤镜”！
            bg = apply_screen_albumentations(bg)

            # 6. 保存图片 (使用连续的 generate_count 命名)
            output_filename = f"fake_{class_id}_{generate_count:04d}.jpg"
            bg.save(os.path.join(out_dir, output_filename), quality=90)

            generate_count += 1  # 计数器+1

        print(f"✅ 类别 [{class_id}] 完成！共合成 {generate_count} 张。")

    print("\n🎉 所有图片分类数据集生成完毕！")


if __name__ == "__main__":
    if not os.path.exists(BASE_OUT_DIR):
        os.makedirs(BASE_OUT_DIR)
    generate_dataset()