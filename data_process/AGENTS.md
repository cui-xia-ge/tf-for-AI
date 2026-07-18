目前在windows系统做数据增强，ubuntu系统训练模型
用于合成的实拍背景图片集："D:\college\718\dataset\box\background"
实拍真实数据集：
windows: "D:\college\718\dataset\box\real"
ubuntu: /home/cgcgs/718/dataset/box/real
合成数据集："D:\college\718\dataset\box\mix"
mix/ 和 real/ 汇总得到总数据集: /home/cgcgs/718/dataset/box/total
上述除实拍真实数据集外的所有图片均已是120*120像素。实拍真实前景应保留原长宽比。
用到了2种合成方法：
1. 高清原图经丰富的数据增强贴到实拍背景： model_train\data_process\generate_fake_dataset.py
2. 裁剪出openart摄像头实拍的原图经简单增强贴到实拍背景：model_train\data_process\generaye_shotmix_dataset.py
