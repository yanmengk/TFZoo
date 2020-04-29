import tensorflow as tf
import tensorflow_datasets as tfds


dataset = tfds.load("mnist", split=tfds.Split.TRAIN)
datset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
datset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)

print(tfds.list_builders()) # 查看 TensorFlow Datasets 当前支持的数据集

