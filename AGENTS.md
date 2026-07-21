## 训练、验证与部署任务规范

本子模块的目标是：在 Ubuntu 上训练10分类箱体模型，选择真实场景和 INT8 指标最优
的模型，并导出可部署到 OpenART 或 MCXN947 MCX 的模型。后续对话处理本目录时，
优先遵守以下约束。

### 评估标准

- 必须同时查看 INT8 test macro-F1、各类召回率、混淆矩阵和实拍准确率，不能只看
  accuracy 或 training loss。
- INT8 与 float 的 accuracy 差值应记录；最终以 INT8 指标和实机结果选模。

### MCX/OpenART 导出约束

- 输入固定为 `120x120x3` RGB，公共接口使用 `[0,255]` 图像；teacher 的 `[-1,1]`
  MobileNetV2 预处理已包含在模型图中。
- 导出必须是完整 INT8，输入和输出 dtype 均为 `int8`，并使用代表性数据校准。
- teacher 的 OpenART 静态检查必须通过，不得包含 Flex/custom operator 或 FLOAT32
  runtime tensor；静态检查不能替代真实 OpenART 加载和推理验证。
- 当前已验证 teacher 约 2.76 MB；这是 OpenART 部署约束，不是 MCXN947 Neutron 的
  student 约束。
- student 转换后执行 `train_model/convert_neutron.py`。默认门槛为
  `NeutronScratch <= 100000` bytes、转换后模型 `<= 430000` bytes；当前 baseline
  scratch 约 86400 bytes。超过门槛时优先缩小 student 或减少激活峰值，不要只看 TFLite
  文件大小。

### Ubuntu TensorFlow/GPU 环境

- 训练必须使用项目 `.venv`，命令优先写成 `uv run ...`，不要使用系统 Python；环境
  创建和排障步骤以根目录 `22.txt` 为准。
- 当前基线为 Python 3.11 和 `tensorflow[and-cuda]==2.21.0`。训练前先确认：
  `uv run python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"`。
- 训练脚本必须在 TensorFlow 初始化后对每块 GPU 调用
  `tf.config.experimental.set_memory_growth(gpu, True)`，避免一次性占满显存。
- 若 GPU 列表为空，优先检查 CUDA/cuDNN 动态库链接；若出现 `ptxas` 错误，补齐
  `.venv/bin/ptxas` 链接后再运行训练。不能把“TensorFlow 已安装”当作 GPU 可用的证明。
- 完成训练环境检查后，再执行本目录的数据补齐、OpenART CNN 训练和 INT8 导出流程。
