import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# 将训练集按照 6:4 的比例进行切割，从而最终我们将得到 15,000个训练样本,
# 10,000 个验证样本以及 25,000 个测试样本
# 文件目录：/Users/yanmk/tensorflow_datasets/imdb_reviews/

train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

print(tf.config.experimental.list_physical_devices("CPU"))

