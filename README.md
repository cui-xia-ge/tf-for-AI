# MCXVision Teacher/Student 模型训练 / Model Training

本子模块用于在 Ubuntu 上训练带标签的十分类箱体模型，并为后续知识蒸馏准备
teacher 和 student。所有类别目录必须按 `00` 到 `09` 命名。

This Ubuntu submodule trains labeled 10-class box classifiers and prepares a
teacher/student pair for later knowledge distillation. Class directories must
be named `00` through `09`.

## 模型分工 / Model Roles

- `teacher`：完整 `MobileNetV2(alpha=1.0, include_top=False)`，优先追求准确率，
  导出后在 OpenART 上运行，并为 student 提供蒸馏知识。
- `student`：从第一层开始降采样的 MCX 小网络，最终转换为 MCXN947 Neutron
  模型，受 Flash 和 SRAM 限制。

- `teacher`: full MobileNetV2 focused on accuracy, OpenART inference, and
  distillation quality.
- `student`: early-downsampling MCX network constrained by MCXN947 Flash and
  SRAM.

训练流程已经移除 MMD、无标签 target batch 和 Focal Loss。`total`、`real`
等通过多次 `--data-dir` 传入的目录都会作为有标签数据参与监督训练。

MMD, unlabeled target batches, and focal loss have been removed. Every
`--data-dir` is treated as labeled supervised data.

训练时每个 `--data-dir` 会在每个类别目录内按 `--seed` 随机抽取
`--validation-fraction`（默认 0.20）作为验证集，剩余图片用于训练。这样可以避免
按目录全局排序切分时验证集只包含最后几个类别的问题；划分数量会写入输出的
`report.json`。

Each labeled directory is split independently and stratified by class. The default
`--validation-fraction 0.20` is reproducible with `--seed`, and the per-class counts
are recorded in `report.json`.

随机文件划分适合当前训练迭代；如果多个合成样本来自同一前景或背景，最终效果仍应
使用未参与合成的独立实拍测试集确认，以避免相关样本造成评估偏乐观。

## 环境安装 / Environment

建议使用 Python 3.10 或 3.11：

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python train_model/test.py
```

如果机器还没有 Python 3.11，先运行 `uv python install 3.11`。`uv` 会创建并管理
项目内的 `.venv`，不需要系统 `pip`。若使用 Python 3.10，只需将上面的版本号改为
`3.10`。

If Python 3.11 is not installed yet, run `uv python install 3.11` first. `uv` manages
the project-local `.venv`, so a system `pip` is not required. For Python 3.10, replace
the version in the command with `3.10`.

The dependency versions are pinned because TensorFlow/Keras checkpoint and
TFLite conversion behavior can change between releases.

### Windows 仅运行数据增强 / Augmentation Only

Windows 上若只运行 `data_process/` 中的数据生成与增强脚本，无需安装 TensorFlow、
scikit-learn 或 TFLite。请使用单独的轻量依赖清单：

```powershell
uv pip install --python .\.venv\Scripts\python.exe `
  -r .\requirements-windows-augmentation.txt
```

This Windows-only list contains NumPy, Albumentations, headless OpenCV, and
Pillow. The original `requirements.txt` remains the complete training
environment for Ubuntu.

## 实拍背景合成 / Shot-on-Background Synthesis

`data_process/generaye_shotmix_dataset.py` 将 OpenART 裁剪前景贴到 120x120
实拍背景。背景会保留原尺寸，并随机增强曝光、对比度、白平衡、gamma、局部对比度和
方向照明；合成后再模拟模糊、噪声、降采样、JPEG 和 RGB565 成像退化。

The script composites cropped OpenART foregrounds onto real 120x120 camera
backgrounds. It preserves background geometry, augments real lighting and
color variation, then applies mild whole-frame camera degradation.

```bash
python data_process/generaye_shotmix_dataset.py \
  --background-dir /home/cgcgs/718/dataset/box/background \
  --foreground-dir /home/cgcgs/718/dataset/box/shot_front \
  --output-dir /home/cgcgs/718/dataset/box/mix \
  --seed 123
```

输出文件固定命名为 `shotmix_{类别}_{序号}.jpg`。使用相同输出目录重跑会直接覆盖
同名文件，不会自动删除本次编号范围之外的历史文件。若需要一个完全干净的数据集，
请在运行前自行清空输出目录。

Output names stay deterministic. A rerun overwrites matching names and does
not delete older files beyond the current index range.

快速预览时可限制每类前景数量；以下两个开关用于消融实拍背景增强和整图成像增强：

```bash
python data_process/generaye_shotmix_dataset.py --max-per-class 4
python data_process/generaye_shotmix_dataset.py --no-background-augmentation
python data_process/generaye_shotmix_dataset.py --no-capture-augmentation
```

## 推荐：ImageNet 预训练 Teacher / Recommended Pretrained Teacher

默认情况下，`train.py --architecture teacher` 会把 `--backbone-weights auto`
解析为 Keras 官方 MobileNetV2 ImageNet-1K 预训练权重。第一次运行会联网下载并
缓存到 `~/.keras/models/`，之后不再重复下载。

By default, teacher `auto` initialization uses the official Keras MobileNetV2
ImageNet-1K weights. Keras downloads them once and caches them under
`~/.keras/models/`.

模型输入是 120x120，而 Keras 官方预训练分辨率列表不包含 120，因此会提示加载
224x224 权重。这是预期提示：`include_top=False` 后卷积核尺寸与输入分辨率无关，
同一组权重可以用于 120x120 输入。

Keras may warn that 120 is not an official pretrained resolution and load the
224x224 weights. This is expected: with `include_top=False`, convolutional
weight shapes do not depend on the input spatial resolution.

```bash
python train_model/train.py \
  --architecture teacher \
  --data-dir /home/cgcgs/718/dataset/box/total \
  --representative-dir /home/cgcgs/718/dataset/box/real \
  --test-dir /home/cgcgs/718/dataset/box/test \
  --head-epochs 5 \
  --epochs 40 \
  --unfreeze-tail-blocks 4 \
  --learning-rate 1e-3 \
  --finetune-learning-rate 1e-5 \
  --output-dir artifacts/teacher
```

首次重训建议显式固定划分参数，例如：

```bash
python train_model/train.py \
  --architecture teacher \
  --data-dir /home/cgcgs/718/dataset/box/total \
  --validation-fraction 0.2 \
  --seed 123 \
  --output-dir artifacts/teacher_stratified
```

采用预训练权重时，程序自动执行两阶段迁移学习：

1. 冻结 MobileNetV2 和 BatchNorm，只训练新建的十分类头。
2. 恢复第一阶段最佳权重，解冻最后若干 MobileNetV2 组，以较小学习率微调；
   BatchNorm 仍保持冻结。

With pretrained weights, training automatically uses two stages: classifier
head warm-up followed by low-learning-rate fine-tuning of the final MobileNetV2
groups. BatchNorm stays frozen during both transfer stages.

### 预训练是否一定更好？ / Is Pretraining Always Better?

通常会更好，尤其当实拍样本不多时。ImageNet 权重已经学习了边缘、纹理、颜色
和局部形状，往往能提高收敛速度和真实场景泛化能力。但本任务包含合成图、固定
箱体和特定摄像头域，收益不能仅凭经验保证。

It is usually better for small and medium datasets because ImageNet features
provide useful edges, textures, colors, and shapes. The synthetic-to-camera
domain gap means the gain must still be measured.

请在固定数据划分和随机种子下至少比较：

```bash
# ImageNet 预训练 / pretrained
python train_model/train.py --architecture teacher --backbone-weights imagenet ...

# 完全随机初始化 / from scratch
python train_model/train.py --architecture teacher --backbone-weights none ...
```

最终按 INT8 测试集 macro-F1、各类别召回率和实拍准确率选择，不按训练 loss
选择。Select by INT8 test macro-F1, per-class recall, and real-camera results.

### Ubuntu 离线使用 / Offline Weights

不能联网时，可先下载与 `alpha=1.0` 匹配的 Keras MobileNetV2 no-top 权重，
然后传入本地文件：

```bash
python train_model/train.py \
  --architecture teacher \
  --backbone-weights /path/to/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5 \
  --data-dir /home/cgcgs/718/dataset/box/total \
  --output-dir artifacts/teacher
```

The local file must be a MobileNetV2 no-top backbone with the same alpha. A
full classifier checkpoint is not interchangeable with a backbone-only file.

## 已有模型微调 / Fine-tune an Existing Checkpoint

`fine-tune.py` 用于继承同架构的完整 `.weights.h5`。它不会覆盖输入文件，仍按
“分类头 -> 末端块”两阶段执行：

```bash
python train_model/fine-tune.py \
  --architecture teacher \
  --pretrained artifacts/teacher/best.weights.h5 \
  --data-dir /home/cgcgs/718/dataset/box/total \
  --representative-dir /home/cgcgs/718/dataset/box/real \
  --output-dir artifacts/teacher_finetuned
```

也可以显式使用 `--pretrained imagenet` 从 Keras ImageNet 权重开始。
Use `--pretrained imagenet` to explicitly start from Keras ImageNet weights.

旧三层 CNN 权重不能直接加载进结构不同的 student；这部分知识必须通过后续蒸馏
继承。The old three-layer CNN cannot directly initialize the structurally
different student; transfer must happen through distillation.

## MCX Student 训练 / Student Training

student 的 `--backbone-weights auto` 自动解析为 `none`，不会错误使用 MobileNet
权重。它保留监督训练、label smoothing、宏 F1 选模、INT8 评估和独立 checkpoint：

```bash
python train_model/train.py \
  --architecture student \
  --data-dir /home/cgcgs/718/dataset/box/total \
  --representative-dir /home/cgcgs/718/dataset/box/real \
  --output-dir artifacts/student

python train_model/fine-tune.py \
  --architecture student \
  --pretrained artifacts/student/best.weights.h5 \
  --data-dir /home/cgcgs/718/dataset/box/total \
  --representative-dir /home/cgcgs/718/dataset/box/real \
  --output-dir artifacts/student_finetuned
```

student 导出后必须通过 Neutron 内存门槛：

```bash
python train_model/convert_neutron.py \
  --converter /path/to/neutron-converter \
  --input artifacts/student_finetuned/box_student_int8.tflite \
  --output artifacts/student_finetuned/box_student_npu.tflite
```

The default gate rejects more than 100,000 bytes of `NeutronScratch` or a
converted student larger than 430,000 bytes. The current student baseline uses
86,400 bytes of scratch.

## OpenART 部署 / OpenART Deployment

teacher 接受 `[0,255]` RGB 输入；MobileNetV2 所需的 `[-1,1]` 归一化已经包含在
量化图中。已验证的全 INT8 teacher 大小约 2.73 MB，仅使用以下 builtin ops：

```text
ADD, MUL, PAD, CONV_2D, DEPTHWISE_CONV_2D,
MEAN, FULLY_CONNECTED, SOFTMAX
```

The export contains no Flex/custom operators or FLOAT32 runtime tensors.

将 `box_teacher_int8.tflite` 复制到 OpenART SD 卡并改名为 `box_cls.tflite`。
现有 `classification.py` 会从 `/sd/box_cls.tflite` 加载并输入 120x120 RGB ROI。
静态检查不能证明具体固件一定有足够 framebuffer，仍需在实机完成加载和一次分类。

Copy `box_teacher_int8.tflite` to the SD card as `box_cls.tflite`. Final
framebuffer capacity and inference must be verified on real OpenART hardware.

## Label Smoothing

默认 `--label-smoothing 0.10` 只是起点，不是已证明最优值。建议固定测试集和 seed，
比较 `0`、`0.05`、`0.10`。十分类且 factor 为 `0.10` 时，正确类目标为 `0.91`，
其余每类为 `0.01`。

The default factor is a starting point. Compare `0`, `0.05`, and `0.10` using
the same split and seed, then select by INT8 test metrics.

每次运行都会生成 `report.json`，包括配置、float/INT8 指标、混淆矩阵、量化参数
和准确率变化。Teacher reports also include the OpenART static operator check.
