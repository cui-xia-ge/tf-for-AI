import tensorflow as tf

# 检查 TensorFlow 是否是用 CUDA（NVIDIA 核心）编译的
is_cuda_built = tf.test.is_built_with_cuda()

# 检查当前运行环境里 GPU 是否真正可用
is_gpu_available = tf.test.is_gpu_available() # 注：较新版本会提示此接口过时，但依然可用

print(f"TensorFlow 编译时是否支持 CUDA: {is_cuda_built}")
print(f"当前运行时 GPU 是否真正可用: {is_gpu_available}")


# 获取当前环境中的物理 GPU 列表
gpus = tf.config.list_physical_devices('GPU')

print("====================================")
if gpus:
    print(f"🎉 恭喜！TensorFlow 成功检测到 GPU！")
    print(f"检测到的 GPU 数量: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f" -> GPU {i}: {gpu}")
else:
    print("❌ 遗憾，未检测到可用的 GPU。目前正在使用 CPU 计算。")
print("====================================")