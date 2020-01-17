import tensorflow as tf
import numpy as np
import argparse

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


parser = argparse.ArgumentParser(description="Process some integers...")
parser.add_argument('--mode', default='train', help='train or test?')
parser.add_argument('--num_epochs', default=1)
parser.add_argument('--batch_size', default=50)
parser.add_argument('--learning_rate', default=0.001)
args = parser.parse_args()

data_loader = MNISTLoader()


def train():
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    num_batches = int(data_loader.num_train_data // args.batch_size * args.num_epochs)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)  # 实例化Checkpoint，设置保存对象为model
    # 使用tf.train.CheckpointManager管理Checkpoint
    manager = tf.train.CheckpointManager(checkpoint, directory='./save', max_to_keep=3)
    for batch_index in range(1, num_batches + 1):
        X, y = data_loader.get_batch(args.batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if batch_index % 100 == 0:  # 每隔100个Batch保存一次
            # 方法一：直接保存
            # path = checkpoint.save('./save/model.ckpt')  # 保存模型参数到文件
            # 方法二：使用CheckpointManager保存模型参数到文件并自定义编号
            path = manager.save(checkpoint_number=batch_index)
            print("model saved to %s" % path)


def test():
    model_to_be_restored = MLP()
    # 实例化Checkpoint，设置恢复对象为新建立的模型model_to_be_restored
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))  # 从文件恢复模型参数
    y_pred = np.argmax(model_to_be_restored.predict(data_loader.test_data), axis=1)
    print("test accuracy: %f" % (sum(y_pred == data_loader.test_label) / data_loader.num_test_data))


if __name__ == "__main__":
    if args.mode == 'train':
        train()
    if args.mode == 'test':
        test()
        # test accuracy: 0.953300
