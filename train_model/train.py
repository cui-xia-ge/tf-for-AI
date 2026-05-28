import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import os
from sklearn.metrics import f1_score
import numpy as np

# ================= 配置区域 =================
DATA_DIR = "D:/college/718/dataset/box/total"  # 数据集根目录
BATCH_SIZE = 16  # 批次大小，CPU训练建议16或32
IMG_SIZE = (120,120)
SEED = 123  # 随机种子，确保训练集和验证集划分不重叠
BEST_MODEL = 'exported model/best_model.h5'
TFLITE = "exported model/exported.tflite"
# ============================================

def load_and_preprocess_dataset():
    print("正在加载训练集...")
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
    print("\n正在加载验证集...")
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
    print(f"\n成功识别到 {len(class_names)} 个类别: {class_names}")

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
    train_ds = train_ds.cache().shuffle(3190).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

def build_mcu_cnn(input_shape=(120,120,3), num_classes=11):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),

        # --- Block 1 ---
        # use_bias=False 是一个小细节：因为后面紧跟着 BatchNorm，
        # BatchNorm 会自己学习偏移量，去掉了 Conv 的 Bias 可以省下一点点内存。
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),  # 输出48x48

        # --- Block 2 ---
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),  # 输出24x24

        # --- Block 3 ---
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),  # 输出12x12

        # --- Output Block ---
        # 将 12x12x64 的特征图拍扁成 1x64 的向量
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        # 输出 10 个类别的 Logits 不加Softmax
        tf.keras.layers.Dense(num_classes)
    ])
    return model


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
            # 模型预测 (因为我们用了 from_logits=True，吐出的是原始得分)
            preds = self.model.predict(images, verbose=0)

            # 使用 argmax 找出得分最高的类别索引
            pred_classes = np.argmax(preds, axis=-1)

            y_pred.extend(pred_classes)
            y_true.extend(labels.numpy())  # 你的标签是 int 编码，直接取 numpy

        # 2. 🚀 调用 sklearn 计算全局真正的 F1-Score
        # average='macro' 是灵魂！它会对所有类别一视同仁求平均。
        # 如果你的皮卡丘只有 10 张，白墙有 1000 张，'macro' 会强迫模型必须把皮卡丘认对，否则分数直接崩盘！
        current_f1 = f1_score(y_true, y_pred, average='macro')
        if logs is not None:
            logs['val_macro_f1'] = current_f1
        # 3. 打印日志并保存模型
        if current_f1 > self.best_f1:
            print(
                f"\nEpoch {epoch + 1:03d}: val_macro_f1 {current_f1:.4f}，保存")
            self.best_f1 = current_f1
            # 只保存权重（推荐），如果想保存整个模型就去掉 save_weights_only
            self.model.save(self.filepath)
        else:
            print(f"\nEpoch {epoch + 1:03d}: val_macro_f1 ({current_f1:.4f})")

def train_model(train_ds, val_ds):
    print("正在构建模型...")
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
    print("\n开始训练...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,  # 可以设大一点，因为有 EarlyStopping 兜底
        callbacks=callbacks
    )
    return model


def export_to_tflite(model, train_ds):
    print("\n开始进行 INT8 全整数量化导出...")
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probability_model.build(input_shape=(None, 120,120,3))
    # 构建代表性数据集生成器 (让量化算法知道你图片的数值分布)
    def representative_data_gen():
        # 从训练集中抽取 100 个批次作为校准数据
        for input_value, _ in train_ds.take(100):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(probability_model)

    # 优化模式
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # 指定代表性数据集
    converter.representative_dataset = representative_data_gen

    # 强制要求所有算子必须转换为 INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # 将模型的输入和输出接口也强制转为 INT8
    # (注意：OpenMV 端输入图像数组时，需要做相应的偏移处理)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8_model = converter.convert()

    # 保存最终的 .tflite 文件
    with open(TFLITE, "wb") as f:
        f.write(tflite_int8_model)

    print(f"模型导出成功！保存在: {TFLITE}")
    print(f"模型体积大小: {os.path.getsize(TFLITE) / 1024:.2f} KB")

if __name__ == "__main__":
    train_dataset, val_dataset, classes = load_and_preprocess_dataset()

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
    model = train_model(train_dataset, val_dataset)
    #model = tf.keras.models.load_model(BEST_MODEL,compile=False)
    export_to_tflite(model, train_dataset)