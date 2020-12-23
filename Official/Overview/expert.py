import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

'''
Subclassing API 提供了由运行定义的高级研究接口。为您的模型创建一个类，然后以命令方式编写前向传播。
您可以轻松编写自定义层、激活函数和训练循环。
'''

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)
# 增加一个维度
x_train = x_train[..., tf.newaxis] # 也可以np.expand_dims(X_train,axis = -1)
x_test = x_test[..., tf.newaxis]

print(x_train.shape)

# 使用tf.data将数据集打乱以及获取成批量数据
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):
    def __init__(self):
        super().__init__()
        # 或者写成 super(MyModel,self).__init__() python2.7的写法
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu') # 或者 units = 128, activation = tf.nn.relu
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# 创建模型的实例
model = MyModel()

# 为训练选择优化器与损失函数：
loss_object = tf.losses.SparseCategoricalCrossentropy()

optimizer = tf.optimizers.Adam()
"""
    tf.losses.SparseCategoricalCrossentropy与tf.losses.sparse_categorical_crossentropy的区别：
        tf.losses.sparse_categorical_crossentropy(y_true=y,y_pred=y_pred)是得到一个batch内每一个样本的loss
            如batch_size=50时，得到的结果是shape=（50，）的Tensor
        而tf.losses.SparseCategoricalCrossentropy()(y_true=y,y_pred=y_pred)得到的是reduce_mean之后的结果  
            此时，结果shape=(),为一个标量，是50个loss值求平均的结果
"""

# 选择衡量指标来度量模型的损失值（loss）和准确率（accuracy）。这些指标在epoch上累积值，然后打印出整体结果。
train_loss = tf.metrics.Mean(name='train_loss')
train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.metrics.Mean(name='test_loss')
test_accuracy = tf.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# 使用tf.GradientTape训练模型
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(y_true=labels, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# 测试模型
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(y_true=labels, y_pred=predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {:.2f}, Accuracy: {:.2f}%, Test Loss: {:.2f}, Test Accuracy: {:.2f}%'
    print(template.format(
        epoch + 1,
        train_loss.result(),
        train_accuracy.result() * 100,
        test_loss.result(),
        test_accuracy.result() * 100
    ))
    # 此模型在测试集上的准确率达到了98.53%
