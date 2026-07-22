目前在windows系统做数据增强，ubuntu系统训练模型
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
- 类别目录必须严格命名为 `00` 到 `09`。训练默认只使用 `total/`；按照当前数据约定，
  `total/` 已由 `mix/` 和 `real/` 汇总得到，不要再次把 `real/` 作为训练目录追加，
  否则会重复采样实拍图片。
- `real/` 只作为 INT8 representative dataset 的默认来源，用于校准，不作为默认训练
  数据。代表性样本默认数量为 500，可用 `--representative-samples` 调整。
- 训练前必须统计 `total/` 和独立 `test/` 的每类数量。任何训练或验证划分缺少类别，
  都不能用总体 accuracy 判断模型；应先补齐数据或调整划分
- 合成脚本输出固定命名 `shotmix_{类别}_{序号}.jpg`，重跑会覆盖同名文件，但不会自动
  删除编号更大的历史文件。不要改成随机文件名或隐式清理，除非用户明确要求。
上述除实拍真实数据集外的所有图片均已是120*120像素。实拍真实前景应保留原长宽比。
用到了2种合成方法：
1. 高清原图经丰富的数据增强贴到实拍背景： model_train\data_process\generate_fake_dataset.py
2. 裁剪出openart摄像头实拍的原图经简单增强贴到实拍背景：model_train\data_process\generaye_shotmix_dataset.py。
