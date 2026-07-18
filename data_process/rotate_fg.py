import os
import glob
from PIL import Image

# ================= 配置区域 =================
BASE_FG_DIR = "D:/college/718/dataset/box/shot_front"  # 前景图片根路径
BASE_OUT_DIR = "D:/college/718/dataset/box/111"  # 输出根路径
CLASSES = [
    ("0", "0"),
    ("1", "1"),
    ("2", "2"),
    ("3", "3"),
    ("4", "4"),
    ("5", "5"),
    ("6", "6"),
    ("7", "7"),
    ("8", "8"),
    ("9", "9")
]


# ============================================

def get_image_paths(directory):
    """获取目录下所有常见格式的图片路径"""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return paths


def rotate_fg_dataset():
    """将前景图片仅顺时针旋转90度并保存到输出目录"""
    for folder_name, class_id in CLASSES:
        fg_dir = os.path.join(BASE_FG_DIR, folder_name)
        out_dir = os.path.join(BASE_OUT_DIR, class_id)

        if not os.path.exists(fg_dir):
            print(f"警告：前景文件夹不存在，跳过 {folder_name}")
            continue

        # 创建输出目录
        os.makedirs(out_dir, exist_ok=True)

        fg_paths = get_image_paths(fg_dir)
        if not fg_paths:
            print(f"警告：{fg_dir} 中没有图片，跳过")
            continue

        print(f"处理类别 [{class_id} - {folder_name}]，共 {len(fg_paths)} 张图片...")
        for fg_path in fg_paths:
            try:
                # 打开图片，自动处理透明通道
                img = Image.open(fg_path)
                # 顺时针旋转90度 (等价于逆时针270度)
                rotated = img.transpose(Image.Transpose.ROTATE_270)
                # 转换为RGB模式，以便保存为JPEG（若原图有透明通道会丢失透明信息，但仅做旋转无需保留）
                if rotated.mode in ('RGBA', 'LA', 'P'):
                    rotated = rotated.convert('RGB')

                # 保留原文件名（不同类别子目录下同名文件不会冲突）
                base_name = os.path.basename(fg_path)
                name, ext = os.path.splitext(base_name)
                # 统一保存为 .jpg 格式，避免格式不一致
                out_path = os.path.join(out_dir, f"{name}.jpg")
                rotated.save(out_path, quality=95)
            except Exception as e:
                print(f"处理失败 {fg_path}: {e}")

        print(f"✅ 类别 {class_id} 完成，已保存至 {out_dir}")

    print("🎉 所有前景图片已顺时针旋转90度并保存完毕！")


if __name__ == "__main__":
    rotate_fg_dataset()