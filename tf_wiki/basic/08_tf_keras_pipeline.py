import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Softmax

# Keras Sequential API
model = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(units=100, activation='relu'),
    Dense(units=10),
    Softmax()
])
model.summary()

# Keras Functional API
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)
model2.summary()

optimizer = tf.optimizers.Adam(learning_rate=0.001)
loss = tf.losses.SparseCategoricalCrossentropy
accuracy = tf.metrics.sparse_categorical_accuracy

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[accuracy])


# 自定义 层
# 如：自定义全连接层
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):  # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_variable(name='w',
                                   shape=[input_shape[-1], self.units], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
                                   shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


# 自定义 损失函数
# 如：自定义均方差损失函数

class MeanSquareError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))


# 自定义 评估指标
# 如：自定义 SparseCategoricalAccuracy

class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name='total',
                                     dtype=tf.int32,
                                     initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count',
                                     dtype=tf.int32,
                                     initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total
