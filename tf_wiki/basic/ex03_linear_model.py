import tensorflow as tf
from tensorflow.keras import Model

import numpy as np

# 上例中简单的线性模型 y_pred = a * X + b ，我们可以通过模型类的方式编写如下
# 方法三：TensorFlow下的线性回归【使用模型类、神经单元】
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)  # 年份
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)  # 房价

# 预处理
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = tf.reshape(X, [5, 1])
y = tf.reshape(y, [5, 1])


class Linear(Model):
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer,
            bias_initializer=tf.zeros_initializer,
        )

    def call(self, input):
        output = self.dense(input)
        return output

# 以下代码结构与前节类似
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
num_epoch = 10000
for i in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(X)  # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)  # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
# a = 0.97637, b = 0.05756506
