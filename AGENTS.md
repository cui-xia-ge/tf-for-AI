用于合成的实拍背景图片集："D:\college\718\dataset\box\background"
实拍真实数据集："D:\college\718\dataset\box\real"
合成数据集："D:\college\718\dataset\box\mix"
mix/ 和 real/ 汇总得到总数据集 total/ 
上述除实拍真实数据集外的所有图片均已是120*120像素。实拍真实前景应保留原长宽比。
用到了2种合成方法：
1. 高清原图经丰富的数据增强贴到实拍背景： model_train\data_process\generate_fake_dataset.py
2. 裁剪出openart摄像头实拍的原图经简单增强贴到实拍背景：model_train\data_process\generaye_shotmix_dataset.py

## 训练、验证与部署任务规范

本子模块的目标是：在 Ubuntu 上训练10分类箱体模型，选择真实场景和 INT8 指标最优
的模型，并导出可部署到 OpenART 或 MCXN947 MCX 的模型。后续对话处理本目录时，
优先遵守以下约束。

### 数据目录与类别

- Windows 原始数据：
  - 背景：`D:\college\718\dataset\box\background`
  - 实拍数据：`D:\college\718\dataset\box\real`
  - 合成数据：`D:\college\718\dataset\box\mix`
  - 训练汇总：`D:\college\718\dataset\box\total`
- Ubuntu 默认路径：`/home/cgcgs/718/dataset/box/total`、
  `/home/cgcgs/718/dataset/box/real`、`/home/cgcgs/718/dataset/box/test`。
- 类别目录必须严格命名为 `00` 到 `09`。训练默认只使用 `total/`；按照当前数据约定，
  `total/` 已由 `mix/` 和 `real/` 汇总得到，不要再次把 `real/` 作为训练目录追加，
  否则会重复采样实拍图片。
- `real/` 只作为 INT8 representative dataset 的默认来源，用于校准，不作为默认训练
  数据。代表性样本默认数量为 500，可用 `--representative-samples` 调整。
- 训练前必须统计 `total/` 和独立 `test/` 的每类数量。任何训练或验证划分缺少类别，
  都不能用总体 accuracy 判断模型；应先补齐数据或调整划分。

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

### 已知日志与代码行为

- `training_lib.py` 的 TFLite 导出阶段已过滤 TensorFlow 转换器打印的内部
  `数字: TensorSpec(shape=(), dtype=tf.resource, name=None)` 行，只保留其它警告和
  量化状态信息。
- `Statistics for quantized inputs were expected...` 和 `tf.lite.Interpreter is deprecated`
  是当前 TensorFlow 版本的警告，不代表导出失败；仍需检查最终 dtype、模型大小和
  OpenART/Neutron 检查结果。
- 合成脚本输出固定命名 `shotmix_{类别}_{序号}.jpg`，重跑会覆盖同名文件，但不会自动
  删除编号更大的历史文件。不要改成随机文件名或隐式清理，除非用户明确要求。
