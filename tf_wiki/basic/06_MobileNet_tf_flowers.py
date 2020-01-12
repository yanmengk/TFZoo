import tensorflow as tf
import tensorflow_datasets as tfds

# num_batches = 1000
# batch_size = 50
# learning_rate = 0.001
#
# dataset = tfds.load('tf_flowers',split=tfds.Split.TRAIN,as_supervised=True)
# dataset = dataset.map(lambda img,label:(tf.image.resize(img,[224,224])/255.0,label)).shuffle(1024).batch(32)
#
# model = tf.keras.applications.MobileNetV2(weights= None,classes = 5)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# loss_object = tf.losses.SparseCategoricalCrossentropy()
#
# for images,labels in dataset:
#     with tf.GradientTape() as tape:
#         labels_pred = model(images)
#         loss = loss_object(y_true=labels,y_pred=labels_pred)
#         print("loss %f " %loss.numpy())
#     grads = tape.gradient(loss,model.trainable_variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads,model.trainable_variables))

'''
    卷积层和池化层工作原理
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPool2D

# TensorFlow 的图像表示为 [图像数目，长，宽，色彩通道数] 的四维张量
# 这里我们的输入图像 image 的张量形状为 [1, 7, 7, 1]
image = np.array([[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 2, 1, 0],
    [0, 0, 2, 2, 0, 1, 0],
    [0, 1, 1, 0, 2, 1, 0],
    [0, 0, 2, 1, 1, 0, 0],
    [0, 2, 1, 1, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]],dtype=np.float32)
print(image.shape) # (1, 7, 7)

image = np.expand_dims(image,axis=-1)
print(image.shape) # (1, 7, 7, 1)

W = np.array([[
    [ 0, 0, -1],
    [ 0, 1, 0],
    [-2, 0, 2]
]],dtype=np.float32)
print(W.shape) # (1, 3, 3)

b = np.array([1], dtype=np.float32)

# 建立一个仅有一个卷积层的模型，用 W 和 b 初始化
model = tf.keras.Sequential([
    Conv2D(
        filters=1, # 卷积层神经元（卷积核）数目
        kernel_size=[3,3], # 感受野W的大小
        kernel_initializer=tf.constant_initializer(W),
        bias_initializer=tf.constant_initializer(b),
        padding='same'
    )
])

output = model(image)
print(tf.squeeze(output))