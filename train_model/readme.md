# OpenART 实拍优先训练说明

## 训练路线

当前主流程不使用 MobileNet、知识蒸馏、MCX student 或 Neutron。训练数据目录为：

```text
/home/cgcgs/718/dataset/box/real/
  00/ ... 09/
```

每类已补齐到 222 张。原始实拍文件按 `split_seed` 固定留出 20% 作为验证集；文件名
以 `shotmix_` 开头的图片全部进入训练集，不进入验证集。这样验证指标只反映真实摄像头
图像，不会被合成图片的相关性高估。

训练流分为两部分：

- 原始实拍训练图：在线执行轻度亮度、对比度和颜色增强。
- ShotMix 训练图：不再叠加新的增强。

增强范围与 `generaye_shotmix_dataset.py` 的前景增强相当，但不包含旋转、翻转、缩放、
模糊、MotionBlur、Gaussian noise、ISO noise、降采样、RGB565 或 JPEG 退化。

INT8 representative dataset 只使用训练部分的原始实拍图。验证报告同时保存 float 和
INT8 accuracy、macro-F1、逐类 recall、混淆矩阵以及量化 accuracy 差值。

## 模型结构

所有候选模型都使用同一结构骨架：

```text
[0,255] RGB, shape=(120,120,3)
  -> Rescaling(1/255)
  -> Conv2D(3x3, stride=2) + BatchNorm + ReLU
  -> Conv2D(3x3, stride=2) + BatchNorm + ReLU
  -> Conv2D(3x3, stride=2) + BatchNorm + ReLU
  -> Conv2D(3x3, stride=2) + BatchNorm + ReLU
  -> Conv2D(3x3, stride=1) + BatchNorm + ReLU
  -> GlobalAveragePooling2D
  -> Dropout
  -> Dense(10)
  -> Softmax（仅导出/评估包装层）
```

四个 stride=2 block 将空间尺寸从 `120x120` 降到 `8x8`。GlobalAveragePooling2D
避免使用大尺寸全连接层，从而降低参数量和过拟合风险。推理图只使用 OpenART 已检查的
builtin 算子：`CONV_2D`、`FULLY_CONNECTED`、`MEAN`、`SOFTMAX`。

### 候选差异

候选只改变每个 block 的通道数和 Dropout，其他训练流程、数据划分和随机 split seed
保持一致。

| 候选 | Block 通道 | Dropout | 参数量 | INT8 文件大小 |
|---|---|---:|---:|---:|
| `narrow` | 16, 32, 64, 96 | 0.30 | 163,898 | 179,104 bytes |
| `medium` | 24, 48, 96, 128 | 0.25 | 313,522 | 331,728 bytes |
| `wide` | 32, 64, 128, 160 | 0.20 | 511,530 | 532,496 bytes |

`narrow` 参数最少、正则化最强；`medium` 在容量和泛化之间折中；`wide` 容量最大但在
当前实拍数据量下更容易过拟合。模型大小上限按现有 OpenART 约 2.76 MB 参考约束检查，
不使用 MCXN947 Neutron 的 scratch/Flash 门槛。

## 训练参数

`train_openart.py` 的默认参数如下：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--batch-size` | 32 | 训练 batch 大小；GPU 正式训练使用 128 |
| `--validation-fraction` | 0.20 | 每类原始实拍留出比例 |
| `--epochs` | 80 | 最大训练轮数 |
| `--patience` | 15 | macro-F1 无提升时的 early stopping 轮数 |
| `--learning-rate` | 0.001 | AdamW 初始学习率，CosineDecay 调度 |
| `--weight-decay` | 0.0001 | AdamW 和卷积/Dense L2 正则 |
| `--label-smoothing` | 0.05 | 交叉熵 label smoothing |
| `--representative-samples` | 500 | INT8 校准实拍样本上限 |
| `--seed` | 123 | 模型初始化和训练随机种子 |
| `--split-seed` | 123 | 所有候选共用的固定实拍划分种子 |
| `--refit-seed` | 456 | 初选候选的第二次初始化确认种子 |

GPU 环境必须按根目录 `22.txt` 和 `AGENTS.md` 使用项目 `.venv`，推荐设置临时 uv 缓存
后运行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python train_model/train_openart.py \
  --data-dir /home/cgcgs/718/dataset/box/real \
  --output-dir artifacts/openart_real \
  --batch-size 128
```

只训练某个候选或执行短跑：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python train_model/train_openart.py \
  --variant narrow --epochs 1 --patience 1 --skip-refit
```

每个候选会先训练并导出完整 INT8，再按以下顺序排名：

1. INT8 实拍验证集 macro-F1；
2. 最低类别召回率；
3. INT8 accuracy；
4. 模型大小。

初选候选会使用 `refit_seed` 在同一 `split_seed` 下复训；只有复训排名不差于初选时才
替换最终模型。

## 当前正式运行结果

本次 GPU 运行使用 `batch_size=128`、`seed=123`、`split_seed=123`。候选结果如下：

| 候选 | float macro-F1 | INT8 macro-F1 | INT8 accuracy | 最低 recall | INT8 大小 |
|---|---:|---:|---:|---:|---:|
| `medium` | 0.952113 | **0.952113** | 0.951872 | 0.866667 | 331,728 |
| `narrow` | 0.957339 | 0.952105 | 0.951872 | 0.866667 | 179,104 |
| `wide` | 0.155262 | 0.155605 | 0.278075 | 0.000000 | 532,496 |

按 INT8 macro-F1 的严格排序，当前发布 `medium`。模型和报告位于：

```text
artifacts/openart_real/best/box_openart_int8.tflite
artifacts/openart_real/best/report.json
artifacts/openart_real/search_report.json
```

`medium` 的第二种训练 seed 在固定划分上明显较差，因此没有替换 seed=123 的初选模型；
这说明后续仍应使用新的独立实拍集和 OpenART 实机结果确认模型稳定性。

## 部署检查

静态检查命令：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python train_model/inspect_openart.py \
  artifacts/openart_real/best/box_openart_int8.tflite
```

静态检查通过只代表 shape、dtype、builtin operator 和 runtime tensor 满足当前约束，
不能替代 OpenART 固件上的真实加载、framebuffer 分配和分类推理验证。
