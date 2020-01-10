import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Activation


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
    num_epochs = 5
    batch_size = 50
    learning_rate = 0.001

    data_loder = MNISTLoader()
    model = MLP()

    # 选择优化器和损失函数
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.losses.SparseCategoricalCrossentropy()
    accuracy_object = tf.metrics.SparseCategoricalAccuracy()

    # 模型训练
    num_batches = int(data_loder.num_train_data // batch_size * num_epochs)  # 6000
    for batch_index in range(num_batches):
        X, y = data_loder.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            train_loss = loss_object(y_true=y,y_pred=y_pred)
            train_accuracy = accuracy_object(y_true=y,y_pred=y_pred)
            print("batch %d: loss: %f, accuracy: %f" % (batch_index, train_loss.numpy(),train_accuracy.numpy()))
        gradients = tape.gradient(train_loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.variables))

    # 模型评估
    # 选择评价函数
    num_batches = int(data_loder.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start, end = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loder.test_data[start:end])
        accuracy_object.update_state(y_true=data_loder.test_label[start:end], y_pred=y_pred)
    print("test accuracy: %f" % accuracy_object.result()) # 0.964052
