import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
    TensorFlow 提供了 tf.data 这一模块，包括了一套灵活的数据集构建 API，
    能够帮助我们快速、高效地构建数据输入的流水线，尤其适用于数据量巨大的场景。
'''

X = tf.constant([2013, 2014, 2015, 2016, 2017])
y = tf.constant([12000, 14000, 15000, 16500, 17500])

# 也可以使用NumPy数组，效果相同
# X = np.array([2013,2014,2015,2016,2017])
# y = np.array([12000,14000,15000,16500,17500])

# dataset = tf.data.Dataset.from_tensor_slices((X,y))
#
# for x,y in dataset:
#     print(x.numpy(),y.numpy())

(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
# train_data = np.expand_dims(train_data.astype(np.float32)/255.0,axis=-1)
train_data = train_data[..., tf.newaxis]
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))


# for image, label in mnist_dataset:
#     plt.title(label.numpy())
#     plt.imshow(image.numpy()[:, :, 0])
#     plt.show()


def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label


# tf.data.Dataset.map示例

# mnist_dataset = mnist_dataset.map(rot90)
#
# for image, label in mnist_dataset:
#     plt.title(label.numpy())
#     plt.imshow(image.numpy()[:, :, 0])
#     plt.show()

# tf.data.Dataset.batch示例

# mnist_dataset = mnist_dataset.batch(4)  # image: [4, 28, 28, 1], labels: [4]
# for images, labels in mnist_dataset:
#     fig, axs = plt.subplots(1, 4)
#     for i in range(4):
#         axs[i].set_title(labels.numpy()[i])
#         axs[i].imshow(images.numpy()[i, :, :, 0])
#     plt.show()

# tf.data.Dataset.shuffle示例

# mnist_dataset = mnist_dataset.shuffle(10000).batch(4)  # 将数据打散后再设置批次，缓存大小设置为 10000
# for images, labels in mnist_dataset:
#     fig, axs = plt.subplots(1, 4)
#     for i in range(4):
#         axs[i].set_title(labels.numpy()[i])
#         axs[i].imshow(images.numpy()[i, :, :, 0])
#     plt.show()


# 使用 Dataset.prefetch() 方法进行数据预加载后的训练流程，
# 在 GPU 进行训练的同时 CPU 进行数据预加载，提高了训练效率。

# mnist_dataset = mnist_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


