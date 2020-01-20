import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]


class MLP(Model):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(100, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    num_batches = 1000
    batch_size = 50
    learning_rate = 0.001
    log_dir = "./tensorboard"

    model = MLP()
    data_loader = MNISTLoader()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    summary_writer = tf.summary.create_file_writer(logdir=log_dir) # 实例化记录器
    tf.summary.trace_on(profiler=True) # 开启Trace（可选）

    for batch_index in range(num_batches):
        X,y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.losses.sparse_categorical_crossentropy(y_true=y,y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f " %(batch_index,loss.numpy()))
            with summary_writer.as_default(): # 指定记录器
                tf.summary.scalar("loss", loss, step=batch_index) # 将当前损失函数的值写入记录器
        grads = tape.gradient(loss,model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace",step=0,profiler_outdir=log_dir) # 保存Trace信息到文件（可选）


