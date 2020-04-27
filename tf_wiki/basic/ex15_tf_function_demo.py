import tensorflow as tf
import numpy as np
import time
from tf_wiki.basic.ex05_CNN_mnist import CNN, MNISTLoader



'''
虽然默认的即时执行模式（Eager Execution）为我们带来了灵活及易调试的特性，但在特定的场合，例如追求高性能或部署模型时，
我们依然希望使用 TensorFlow 1.X 中默认的图执行模式（Graph Execution），将模型转换为高效的 TensorFlow 图模型。
此时，TensorFlow 2 为我们提供了 tf.function 模块，结合 AutoGraph 机制，使得我们仅需加入一个简单的 @tf.function 修饰符，
就能轻松将模型以图执行模式运行。
'''

num_batches = 400
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()
model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# @tf.function
# def train_one_step(X, y):
#     with tf.GradientTape() as tape:
#         y_pred = model(X)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=y_pred)
#         loss = tf.reduce_mean(loss)
#         tf.print("loss",loss) # 这里使用了TensorFlow内置的tf.print()。@tf.function不支持Python内置的print方法
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
#
#
# start_time = time.time()
# for batch_index in range(num_batches):
#     X, y = data_loader.get_batch(batch_size)
#     train_one_step(X, y)
# end_time = time.time()
# print(end_time - start_time)


@tf.function
def f(x):
    print("The function is running in Python")
    tf.print(x)


a = tf.constant(1, dtype=tf.int32)
f(a)
b = tf.constant(2, dtype=tf.int32)
f(b)
b_ = np.array(2, dtype=np.int32)
f(b_)
c = tf.constant(0.1, dtype=tf.float32)
f(c)
d = tf.constant(0.2, dtype=tf.float32)
f(d)

