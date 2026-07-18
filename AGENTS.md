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