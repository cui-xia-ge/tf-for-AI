import os
# 必须在导入 tf 之前设置！
# 告诉 TensorFlow 底层：忽略Info和Warning，不然会被刷屏
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score
import numpy as np


# ================= 配置区域 =================
DATA_DIR = "/home/cgcgs/718/dataset/box/total"  # 数据集根目录
REAL = "/home/cgcgs/718/dataset/box/real"       # 实拍照片
BATCH_SIZE = 16  # 批次大小，CPU训练建议16或32
IMG_SIZE = (120,120)
SEED = 123  # 随机种子，确保训练集和验证集划分不重叠
BEST_MODEL = 'exported model/best_model.weights.h5'
TFLITE = "exported model/exported.tflite"
# ============================================

def load_and_preprocess_dataset():
    print("正在加载训练集...\n")
    # 1. 加载训练集 (自动从文件夹名称 0-9 提取标签)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,  # 划出 20% 作为验证集
        subset="training",  # 指定这是训练部分
        seed=SEED,
        color_mode="rgb",  # 极限优化：直接在读取时转为灰度图 (1个通道)
        image_size=IMG_SIZE,  # 缩放图片
        batch_size=BATCH_SIZE,
        label_mode='int'  # 'int' 对应整数标签 0-9
    )
    print("正在加载验证集...\n")
    # 2. 加载验证集
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",  # 指定这是验证部分
        seed=SEED,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    # 获取类别名称 (应该是 ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    class_names = train_ds.class_names
    print(f"成功识别到 {len(class_names)} 个类别: {class_names}\n")

    #不归一化换取更稳健的预测！ BN层会归一化 + 调小lr + 调大epoch 可以减少训练初期梯度过大、震荡的影响。
    '''# 3. 数据预处理管道 (归一化到 0 ~ 1)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    # 映射归一化操作
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)'''

    # 4. 性能优化 (极大地加快训练速度)
    # cache() 将数据缓存在内存中，避免每个 epoch 都去读硬盘
    # shuffle() 就地打乱数据
    # prefetch() 让 CPU 在 GPU/CPU 训练当前批次时，后台提前准备好下一个批次
    train_ds = train_ds.cache().shuffle(3000).prefetch(buffer_size=4)
    val_ds = val_ds.cache().prefetch(buffer_size=4)

    return train_ds, val_ds, class_names

def build_mcu_cnn(input_shape=(120,120,3), num_classes=10):
    # 1.用 Sequential 封装纯线性的主干网络 (特征提取器)
    backbone = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Block 2 
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Block 3 
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # 汇聚层 
        tf.keras.layers.GlobalAveragePooling2D()
    ], name="mcu_cnn_backbone")

    # 2. 💡 使用 Functional API 实现分叉输出
    inputs = tf.keras.Input(shape=input_shape)
    
    # 让输入流过主干网络，得到基础高维特征
    backbone_output = backbone(inputs)
    
    # 分叉点 1：拦截 Dropout 后的特征给 MMD 使用
    features = tf.keras.layers.Dropout(0.4)(backbone_output)
    
    # 分叉点 2：分类头得到 Logits
    logits = tf.keras.layers.Dense(num_classes)(features)
    
    # 组装完整的双输出模型
    model = tf.keras.Model(inputs=inputs, outputs=[features, logits])
    return model

def compute_mmd(source_features, target_features, sigmas=[1.0, 5.0, 10.0]):
    """
    计算多核 MMD (Multi-Kernel Maximum Mean Discrepancy)
    source_features: 源域(合成)特征, shape (batch_size, feature_dim)
    target_features: 目标域(实拍)特征, shape (batch_size, feature_dim)
    """
    # 动态获取当前的 batch_size
    n_s = tf.shape(source_features)[0]
    n_t = tf.shape(target_features)[0]

    # 合并特征以便进行矩阵运算 (n_s + n_t, feature_dim)
    features = tf.concat([source_features, target_features], axis=0)
    
    # 计算两两之间的欧氏距离的平方 ||x - y||^2
    # 利用展开公式: (x-y)^2 = x^2 + y^2 - 2xy
    xx = tf.matmul(features, features, transpose_b=True)
    rx = tf.broadcast_to(tf.expand_dims(tf.linalg.diag_part(xx), 1), tf.shape(xx))
    ry = tf.broadcast_to(tf.expand_dims(tf.linalg.diag_part(xx), 0), tf.shape(xx))
    distance_sq = rx + ry - 2.0 * xx

    # 计算高斯多核
    kernel_val = tf.zeros_like(distance_sq)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma ** 2)
        kernel_val += tf.exp(-gamma * distance_sq)

    # 划分核矩阵
    # K_ss: 源域内, K_tt: 目标域内, K_st: 源域与目标域间
    K_ss = kernel_val[:n_s, :n_s]
    K_tt = kernel_val[n_s:, n_s:]
    K_st = kernel_val[:n_s, n_s:]

    # 按照 MMD 公式求和并平均
    mmd = tf.reduce_mean(K_ss) + tf.reduce_mean(K_tt) - 2.0 * tf.reduce_mean(K_st)
    # 返回非负值（防止浮点误差导致极小的负数）
    return tf.maximum(mmd, 0.0)

class MMD_DomainAdaptationModel(tf.keras.Model):
    def __init__(self, cnn_extractor, mmd_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.cnn = cnn_extractor
        self.mmd_weight = mmd_weight # MMD 损失占总损失的比例 (lambda)
        
        # 定义需要追踪的指标
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.cls_loss_tracker = tf.keras.metrics.Mean(name="cls_loss")
        self.mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.cls_loss_tracker, self.mmd_loss_tracker]

    def call(self, inputs, training=False):
        # 兼容普通的 predict 和 evaluate 操作
        return self.cnn(inputs, training=training)

    def train_step(self, data):
        # ⚠️ 注意：传进来的 data 是打包好的 ((源域图, 源域标签), 目标域图)
        (source_x, source_y), target_x = data

        with tf.GradientTape() as tape:
            # 1. 源域过一遍网络
            source_features, source_logits = self.cnn(source_x, training=True)
            # 2. 目标域过一遍网络
            target_features, _ = self.cnn(target_x, training=True)

            # 3. 计算源域的分类 Loss (调用你之前手写的 Focal Loss)
            cls_loss = self.compiled_loss(source_y, source_logits)

            # 4. 计算两者的 MMD 距离
            mmd_loss = compute_mmd(source_features, target_features)

            # 5. 总 Loss = 分类 Loss + lambda * MMD Loss
            total_loss = cls_loss + (self.mmd_weight * mmd_loss)

        # 反向传播并更新权重
        gradients = tape.gradient(total_loss, self.cnn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.cnn.trainable_variables))

        # 更新日志进度条
        self.total_loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.mmd_loss_tracker.update_state(mmd_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "cls_loss": self.cls_loss_tracker.result(),
            "mmd_loss": self.mmd_loss_tracker.result(),
        }
    
    def test_step(self, data):
        # 验证时只需要算分类 Loss (目标域没标签，无法验证)
        x, y = data
        _, logits = self.cnn(x, training=False)
        cls_loss = self.compiled_loss(y, logits)
        return {"loss": cls_loss}
    
class CustomSparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super(CustomSparseCategoricalFocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # ================= 1. 安全计算基础 Sparse 交叉熵 =================
        if self.from_logits:
            # 💡 核心修改 1：调用 sparse_categorical_crossentropy 替代原本的接口
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
            pred_prob = tf.nn.softmax(y_pred, axis=-1)
        else:
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
            pred_prob = y_pred

        # ================= 2. 提取 p_t (真实类别的预测概率) =================
        # 💡 核心修改 2：把整数标签 (如 3) 临时转换成独热编码 (如 [0,0,0,1,0...])
        # 获取类别总数
        num_classes = tf.shape(y_pred)[-1]

        # 将 y_true 强制展平并转为整型，防止传入形状为 (batch, 1) 时引发广播错误
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)

        # 将 pred_prob 也展平到二维 (batch, classes)，确保矩阵维度绝对对齐
        pred_prob_flat = tf.reshape(pred_prob, [-1, num_classes])

        # 提取目标概率 p_t
        p_t = tf.reduce_sum(y_true_one_hot * pred_prob_flat, axis=-1)

        # ================= 3. 计算 Focal 动态缩放权重 =================
        weight = self.alpha * tf.math.pow((1.0 - p_t), self.gamma)

        # ================= 4. 融合并输出 =================
        # 将 ce_loss 也展平，防止维度报错
        ce_loss = tf.reshape(ce_loss, [-1])
        focal_loss = weight * ce_loss

        return K.mean(focal_loss, axis=-1)


class F1ScoreCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, filepath):
        super().__init__()
        self.val_dataset = val_dataset
        self.filepath = filepath
        self.best_f1 = 0.0  # 记录历史最高 F1

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []

        # 1. 遍历整个验证集进行推理
        for images, labels in self.val_dataset:
            # 🚀 极限优化：直接作为可调用对象执行前向传播，跳过 predict 的底层开销！
            # 注意：因为你加了 MMD，现在的模型输出是 [features, logits]，所以要取 [1]
            preds = self.model(images, training=False)[1] 
            
            # 由于输出的是 TF 张量，用 tf.argmax 算完后再转回 numpy
            pred_classes = tf.argmax(preds, axis=-1).numpy()
            
            y_pred.extend(pred_classes)
            y_true.extend(labels.numpy())

        # 2. 🚀 调用 sklearn 计算全局真正的 F1-Score
        # average='macro' 是灵魂！它会对所有类别一视同仁求平均。
        # 如果你的皮卡丘只有 10 张，白墙有 1000 张，'macro' 会强迫模型必须把皮卡丘认对，否则分数直接崩盘！
        current_f1 = f1_score(y_true, y_pred, average='macro')
        if logs is not None:
            logs['val_macro_f1'] = current_f1
        # 3. 打印日志并保存模型
        if current_f1 > self.best_f1:
            print(f"Epoch {epoch + 1:03d}: val_macro_f1 {current_f1:.4f}，保存\n")
            self.best_f1 = current_f1
            # 🚀 改为只保存权重！彻底避开模型结构的序列化问题
            self.model.save_weights(self.filepath)
        else:
            print(f"Epoch {epoch + 1:03d}: val_macro_f1 ({current_f1:.4f})\n")

def train_model(train_ds, val_ds):
    print("正在构建模型...\n")
    model = build_mcu_cnn()
    model.summary()  # 打印网络结构和参数量
    # 计算总的训练步数 (Steps)
    # BATCH_SIZE = 16, 假设你有 800 张训练图片，那么一个 Epoch 有 800/16 = 50 个 Step
    # 你可以通过 len(train_ds) 直接获取一个 Epoch 的 Step 数量
    epochs = 70
    steps_per_epoch = len(train_ds)
    total_decay_steps = steps_per_epoch * epochs
    # 余弦退火调度器
    # 初始学习率给 0.001，在 total_decay_steps 内，以余弦曲线的形态慢慢降到 alpha * 0.001 (即 0.00001)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=total_decay_steps,
        alpha=0.01  # 最终学习率是初始学习率的 1%
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # 编译模型

    model.compile(
        optimizer=optimizer,
        loss=CustomSparseCategoricalFocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    '''loss=tf.keras.losses.CategoricalFocalCrossentropy(
        alpha=0.25,  # 类别权重，可以用列表为每个类别单独设置，也可以统一设一个小数
        gamma=2.0,   # 聚焦参数，2.0 是黄金默认值
        from_logits=False # 如果你的模型最后一层有 Softmax，这里设为 False
    ),'''

    # 设置回调函数
    callbacks = [
        # 保存验证集准确率最高的model
        # tf.keras.callbacks.ModelCheckpoint(filepath=BEST_MODEL, monitor='val_accuracy', save_best_only=True)
        F1ScoreCheckpoint(val_dataset=val_dataset, filepath=BEST_MODEL),
        tf.keras.callbacks.EarlyStopping(monitor='val_macro_f1',  # 在执行F1ScoreCheckpoint时写入log的key
        mode='max', patience=15, restore_best_weights=True),
    ]
    print("开始训练...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,  # 可以设大一点，因为有 EarlyStopping 兜底
        callbacks=callbacks
    )
    return model


def export_to_tflite(model, train_ds):
    print("开始进行针对单片机优化的 INT8 全整数量化导出...\n")
    # ⚠️ 核心解耦魔法：
    # 此时传入的 model 是训练好的双输出 base_cnn。
    # model.inputs 获取模型的输入接口。
    # model.outputs[1] 精准提取出索引为 1 的 logits 分类输出分支！
    inference_only_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs[1])
    
    # 在分类分支后面，缝合上单片机推理需要的 Softmax 激活函数
    probability_model = tf.keras.Sequential([
        inference_only_model,
        tf.keras.layers.Softmax()
    ])
    probability_model.build(input_shape=(None, 120, 120, 3))
    
    # 2. 构建代表性数据集生成器 (校准 INT8 精度)
    def representative_data_gen():
        for input_value, _ in train_ds.take(100):
            yield [input_value]

    # 3. 启动转换器 (此时转换器看到的是纯净的单输入、单输出模型了！)
    converter = tf.lite.TFLiteConverter.from_keras_model(probability_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 强制要求输入输出为整型，完美贴合 OpenMV 摄像头
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8_model = converter.convert()

    # 保存最终的 .tflite 文件
    with open(TFLITE, "wb") as f:
        f.write(tflite_int8_model)

    print(f"模型导出成功！保存在: {TFLITE}\n")
    print(f"模型体积大小: {os.path.getsize(TFLITE) / 1024:.2f} KB\n")

if __name__ == "__main__":
    train_source_ds, val_ds, classes = load_and_preprocess_dataset()

    """ # 取出一个 Batch 的数据来看看
    for images, labels in train_dataset.take(1):
        print(f"\n图像 Batch 形状: {images.shape}")  # 预期: (16, 96, 96, 1)
        print(f"标签 Batch 形状: {labels.shape}")  # 预期: (16,)
        print(f"像素值范围: min={tf.reduce_min(images):.2f}, max={tf.reduce_max(images):.2f}")  # 预期: 0.0 ~ 1.0

        # 画出前 9 张图验证一下
        plt.figure(figsize=(8, 8))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            # 因为是灰度图 (..., 1)，imshow 需要去掉最后一个通道维度
            plt.imshow(tf.squeeze(images[i]), cmap='gray')
            plt.title(f"Label: {classes[labels[i]]}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()"""
    
    # 2. 🚀 新增：加载无标签的实拍图片 (目标域)
    target_ds = tf.keras.utils.image_dataset_from_directory(
        REAL,                    # 实拍照片目录
        labels=None,             # 没有标签！
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    # 使用 repeat() 让目标域数据无限循环，以对齐源域数据的数据量
    target_ds = target_ds.repeat().prefetch(buffer_size=4)

    # 3. 🚀 灵魂一步：用 zip 把它们打包在一起
    # 产出格式: ((source_img, source_label), target_img)
    da_train_dataset = tf.data.Dataset.zip((train_source_ds, target_ds)).prefetch(buffer_size=4)

    # 4. 构建网络与 DA 壳子
    base_cnn = build_mcu_cnn()
    da_model = MMD_DomainAdaptationModel(base_cnn, mmd_weight=0.2) # lambda 设置为 0.2

    base_cnn.summary()  # 打印网络结构和参数量
    # 计算总的训练步数 (Steps)
    # BATCH_SIZE = 16, 假设你有 800 张训练图片，那么一个 Epoch 有 800/16 = 50 个 Step
    # 你可以通过 len(train_ds) 直接获取一个 Epoch 的 Step 数量
    epochs = 70
    steps_per_epoch = len(train_source_ds)
    total_decay_steps = steps_per_epoch * epochs
    # 余弦退火调度器
    # 初始学习率给 0.001，在 total_decay_steps 内，以余弦曲线的形态慢慢降到 alpha * 0.001 (即 0.00001)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=total_decay_steps,
        alpha=0.01  # 最终学习率是初始学习率的 1%
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # 设置回调函数
    callbacks = [
        # 保存验证集准确率最高的model
        # tf.keras.callbacks.ModelCheckpoint(filepath=BEST_MODEL, monitor='val_accuracy', save_best_only=True)
        F1ScoreCheckpoint(val_dataset=val_ds, filepath=BEST_MODEL),
        tf.keras.callbacks.EarlyStopping(monitor='val_macro_f1',  # 在执行F1ScoreCheckpoint时写入log的key
        mode='max', patience=15, restore_best_weights=True),
    ]
    da_model.compile(
        optimizer=optimizer,
        loss=CustomSparseCategoricalFocalLoss(gamma=2.0, alpha=0.25, from_logits=True)
        #metrics=['accuracy']
    )

    # 开始炼丹！
    da_model.fit(
        da_train_dataset,
        validation_data=val_ds, # 验证集不变
        epochs=epochs,
        steps_per_epoch=len(train_source_ds), # 必须指定 steps，因为 target_ds 是无限循环的
        callbacks=callbacks,
        verbose=2   #不要动态进度条，简化输出
    )
    export_to_tflite(da_model.cnn, train_source_ds)