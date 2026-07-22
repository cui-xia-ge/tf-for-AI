你写的代码要有关键处的中文注释
每完成一个任务进行提交
## 训练、验证与部署任务规范

本目录用于在 Ubuntu 上训练箱体分类模型，并导出可部署模型。后续处理本目录时，优先遵守以下约束。

### GPU 训练环境

- 训练必须使用项目根目录 `.venv`，命令优先写成 `uv run ...`，不要使用系统 Python。
- 环境创建和排障步骤以根目录 `22.txt` 为准；不要仅凭 TensorFlow 已安装就判断 GPU 可用。
- 当前推荐基线为 Python 3.11 和 `tensorflow[and-cuda]==2.21.0`。
- 训练前必须先确认 Python、TensorFlow 和 GPU 可见性：

```bash
uv run python -c "import sys,tensorflow as tf; print(sys.executable); print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

- 正常情况下应看到 `.venv` 下的 Python、TensorFlow 2.21.0，以及非空 GPU 列表。
- 如果 GPU 列表为空，优先检查 CUDA/cuDNN 动态库链接。可按 `22.txt` 在 TensorFlow 包目录下补链接：

```bash
pushd $(dirname $(python -c 'import tensorflow as tf; print(tf.__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd
```

- 如果出现 `ptxas` 相关错误，补齐 `.venv/bin/ptxas` 链接后再运行训练：

```bash
ln -sf $(find .venv/lib -path "*/nvidia/cuda_nvcc/*/bin/ptxas" -print -quit) .venv/bin/ptxas
```

- 训练脚本必须在 TensorFlow 初始化后对每块 GPU 调用显存按需增长，避免一开始占满显存：

```python
import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("GPUs:", tf.config.list_physical_devices("GPU"))
```

- 完成 GPU 环境检查后，再执行数据补齐、模型训练、INT8 导出和部署验证流程。
- 若出现 `Cannot dlopen some GPU libraries`，说明 CUDA/cuDNN 动态库仍未被 TensorFlow 找到，应重新按 `22.txt` 补动态库链接。
- 若出现 `Visible GPUs: 0`，先确认命令实际使用的是 `.venv`：

```bash
uv run python -c "import sys; print(sys.executable)"
```

- 若出现 `no kernel image is available`、`unsupported PTX` 或 `device kernel image is invalid`，优先参考 `22.txt` 的 NVIDIA TensorFlow 容器方案。

### 评估标准

- 选模不能只看 accuracy 或 training loss；必须同时查看 INT8 test macro-F1、各类召回率、混淆矩阵和实拍准确率。
- 必须记录 INT8 与 float 的 accuracy 差值。
- 最终以 INT8 指标和真实设备或真实场景结果选模。
