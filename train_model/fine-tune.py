import os
# 告诉 TensorFlow 底层：忽略Info和Warning，保持终端清爽
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
BATCH_SIZE = 32  # 批次大小
IMG_SIZE = (120, 120)
SEED = 123  
BEST_MODEL = 'exported model/best_model.weights.h5'
TFLITE = "exported model/exported.tflite"
# ============================================

def load_and_preprocess_dataset():
    print("正在加载训练集...\n")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    print("正在加载验证集...\n")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    class_names = train_ds.class_names
    print(f"成功识别到 {len(class_names)} 个类别: {class_names}\n")

    train_ds = train_ds.cache().shuffle(5000).prefetch(buffer_size=4)
    val_ds = val_ds.cache().prefetch(buffer_size=4)

    return train_ds, val_ds, class_names

def build_mcu_cnn(input_shape=(120, 120, 3), num_classes=10):
    # 🚀 新增：构建极其克制的数据增强模块 (只在 training=True 时生效)
    # 我们只允许正负 10% 的缩放，防止数字 (如 6 和 9) 或者箱子特征严重变形
    data_augmentation = tf.keras.Sequential([
        # height_factor 和 width_factor 设为 0.1 表示随机放大或缩小最多 10%
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='constant', fill_value=0.0) 
    ], name="data_augmentation")
    backbone = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        # 🚀 新增：将增强模块作为网络的第一道关卡
        data_augmentation,
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

    inputs = tf.keras.Input(shape=input_shape)
    backbone_output = backbone(inputs)
    features = tf.keras.layers.Dropout(0.25)(backbone_output)
    logits = tf.keras.layers.Dense(num_classes)(features)
    
    model = tf.keras.Model(inputs=inputs, outputs=[features, logits])
    return model

def compute_mmd(source_features, target_features):
    """动态多核 MMD"""
    n_s = tf.shape(source_features)[0]
    n_t = tf.shape(target_features)[0]

    features = tf.concat([source_features, target_features], axis=0)
    xx = tf.matmul(features, features, transpose_b=True)
    rx = tf.broadcast_to(tf.expand_dims(tf.linalg.diag_part(xx), 1), tf.shape(xx))
    ry = tf.broadcast_to(tf.expand_dims(tf.linalg.diag_part(xx), 0), tf.shape(xx))
    distance_sq = rx + ry - 2.0 * xx

    bandwidth = tf.stop_gradient(tf.reduce_mean(distance_sq))
    bandwidth = tf.maximum(bandwidth, 1e-5) 

    kernel_val = tf.zeros_like(distance_sq)
    multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
    for multiplier in multipliers:
        gamma = 1.0 / (2.0 * bandwidth * multiplier)
        kernel_val += tf.exp(-gamma * distance_sq)

    K_ss = kernel_val[:n_s, :n_s]
    K_tt = kernel_val[n_s:, n_s:]
    K_st = kernel_val[:n_s, n_s:]

    mmd = tf.reduce_mean(K_ss) + tf.reduce_mean(K_tt) - 2.0 * tf.reduce_mean(K_st)
    return tf.maximum(mmd, 0.0)

class MMD_DomainAdaptationModel(tf.keras.Model):
    def __init__(self, cnn_extractor, mmd_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.cnn = cnn_extractor
        self.mmd_weight = mmd_weight 
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.cls_loss_tracker = tf.keras.metrics.Mean(name="cls_loss")
        self.mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.cls_loss_tracker, self.mmd_loss_tracker]

    def call(self, inputs, training=False):
        return self.cnn(inputs, training=training)

    def train_step(self, data):
        (source_x, source_y), target_x = data

        with tf.GradientTape() as tape:
            source_features, source_logits = self.cnn(source_x, training=True)
            target_features, _ = self.cnn(target_x, training=True)

            cls_loss = self.compiled_loss(source_y, source_logits)
            mmd_loss = compute_mmd(source_features, target_features) 
            total_loss = cls_loss + (self.mmd_weight * mmd_loss)

        gradients = tape.gradient(total_loss, self.cnn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.cnn.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.mmd_loss_tracker.update_state(mmd_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "cls_loss": self.cls_loss_tracker.result(),
            "mmd_loss": self.mmd_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x, y = data
        _, logits = self.cnn(x, training=False)
        cls_loss = self.compiled_loss(y, logits)
        return {"loss": cls_loss}
    
class CustomSparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=1.0, from_logits=True, **kwargs):
        super(CustomSparseCategoricalFocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha 
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
            pred_prob = tf.nn.softmax(y_pred, axis=-1)
        else:
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
            pred_prob = y_pred

        num_classes = tf.shape(y_pred)[-1]
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)

        pred_prob_flat = tf.reshape(pred_prob, [-1, num_classes])
        p_t = tf.reduce_sum(y_true_one_hot * pred_prob_flat, axis=-1)

        if isinstance(self.alpha, (list, np.ndarray)):
            alpha_tensor = tf.convert_to_tensor(self.alpha, dtype=tf.float32)
            alpha_factor = tf.gather(alpha_tensor, y_true_int)
        else:
            alpha_factor = self.alpha 

        weight = alpha_factor * tf.math.pow((1.0 - p_t), self.gamma)
        ce_loss = tf.reshape(ce_loss, [-1])
        focal_loss = weight * ce_loss
        return K.mean(focal_loss, axis=-1)

class F1ScoreCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, filepath):
        super().__init__()
        self.val_dataset = val_dataset
        self.filepath = filepath
        self.best_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []

        for images, labels in self.val_dataset:
            preds = self.model(images, training=False)[1] 
            pred_classes = tf.argmax(preds, axis=-1).numpy()
            y_pred.extend(pred_classes)
            y_true.extend(labels.numpy())

        current_f1 = f1_score(y_true, y_pred, average='macro')
        if logs is not None:
            logs['val_macro_f1'] = current_f1
            
        if current_f1 > self.best_f1:
            print(f"Epoch {epoch + 1:03d}: val_macro_f1 {current_f1:.4f}，保存\n")
            self.best_f1 = current_f1
            self.model.save_weights(self.filepath)
        else:
            print(f"Epoch {epoch + 1:03d}: val_macro_f1 ({current_f1:.4f})\n")

def export_to_tflite(model, train_ds):
    print("开始进行针对单片机优化的 INT8 全整数量化导出...\n")
    inference_only_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs[1])
    
    probability_model = tf.keras.Sequential([
        inference_only_model,
        tf.keras.layers.Softmax()
    ])
    probability_model.build(input_shape=(None, 120, 120, 3))
    
    def representative_data_gen():
        for input_value, _ in train_ds.take(100):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(probability_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8_model = converter.convert()

    with open(TFLITE, "wb") as f:
        f.write(tflite_int8_model)

    print(f"模型导出成功！保存在: {TFLITE}\n")
    print(f"模型体积大小: {os.path.getsize(TFLITE) / 1024:.2f} KB\n")

if __name__ == "__main__":
    # --- 1. 数据集准备 ---
    train_source_ds, val_ds, class_names = load_and_preprocess_dataset()
    num_classes = len(class_names)
    
    target_ds = tf.keras.utils.image_dataset_from_directory(
        REAL, labels=None, color_mode="rgb", image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    target_ds = target_ds.repeat().prefetch(buffer_size=4)
    da_train_dataset = tf.data.Dataset.zip((train_source_ds, target_ds)).prefetch(buffer_size=4)

    # --- 2. 构建模型 ---
    base_cnn = build_mcu_cnn(num_classes=num_classes)
    
    # 🚀 应对“类别不平衡的域偏移”：大幅降低 MMD 权重 (如 0.05)，防止特征被强行扭曲
    da_model = MMD_DomainAdaptationModel(base_cnn, mmd_weight=0.05) 

    # --- 3. 🚀 加载预训练权重 (Fine-tune 核心逻辑) ---
    print("\n================== 权重加载 ==================")
    # 必须先执行 build 或过一遍 dummy 数据，Keras 才会把底层的网络参数对象建立起来
    da_model.build(input_shape=(None, 120, 120, 3))
    
    # 检查你之前保存的模型文件是否存在
    if os.path.exists(BEST_MODEL):
        print(f"找到预训练权重文件: {BEST_MODEL}")
        da_model.load_weights(BEST_MODEL)
        print("权重加载成功！进入微调(Fine-tune)模式。")
    else:
        print(f"未找到预训练权重 {BEST_MODEL}，将从头开始(Scratch)训练。")
    print("==============================================\n")
    
    base_cnn.summary() 

    # --- 4. 配置微调超参数 ---
    epochs = 40  # 微调不需要太多 Epoch
    steps_per_epoch = len(train_source_ds)
    total_decay_steps = steps_per_epoch * epochs

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,  # 🚀 微调专属：初始学习率暴降 10 倍 (0.0001)，保护既有特征
        decay_steps=total_decay_steps,
        alpha=0.01  
    )
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    callbacks = [
        F1ScoreCheckpoint(val_dataset=val_ds, filepath=BEST_MODEL),
        tf.keras.callbacks.EarlyStopping(monitor='val_macro_f1', 
        mode='max', patience=20, restore_best_weights=True), # 微调的容忍度也可稍微缩短
    ]
    
    da_model.compile(
        optimizer=optimizer,
        loss=CustomSparseCategoricalFocalLoss(gamma=2.0, alpha=1.0, from_logits=True)
    )

    # --- 5. 开始训练与导出 ---
    da_model.fit(
        da_train_dataset,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=len(train_source_ds), 
        callbacks=callbacks,
        verbose=2
    )
    
    export_to_tflite(da_model.cnn, train_source_ds)