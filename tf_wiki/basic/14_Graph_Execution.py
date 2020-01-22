import tensorflow as tf
import numpy as np
import time

from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Activation, Conv2D, MaxPool2D, Reshape

'''
    在 TensorFlow 2.0 中，推荐使用 @tf.function （而非 1.X 中的 tf.Session ）实现 Graph Execution，
    从而将模型转换为易于部署且高性能的 TensorFlow 图模型。只需要将我们希望以 Graph Execution 模式运行的代码封装
    在一个函数内，并在函数前加上 @tf.function 即可
'''

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


class CNN(Model):
    def __init__(self):
        super().__init__()
        # self.flatten = Flatten()
        # self.dense1 = Dense(100, activation='relu')
        # self.dense2 = Dense(10, activation='softmax')
        self.conv1 = Conv2D(
            filters=32,  # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation='relu'
        )
        self.pool1 = MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = Conv2D(
            filters=64,  # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation='relu'
        )
        self.pool2 = MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = Dense(units=1024, activation='relu')
        self.dense2 = Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return tf.nn.softmax(x)



num_batches = 400
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()


@tf.function
def train_one_step(X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        # 注意这里使用了TensorFlow内置的tf.print()。@tf.function不支持Python内置的print方法
        tf.print("loss", loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


if __name__ == '__main__':
    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    start_time = time.time()
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        train_one_step(X, y)
    end_time = time.time()
    print(end_time - start_time)