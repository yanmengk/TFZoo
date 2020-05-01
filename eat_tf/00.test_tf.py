import tensorflow as tf

tf.print("TensorFlow version: ", tf.__version__)

a = tf.constant("hello")
b = tf.constant("tensorflow2")
c = tf.strings.join([a, b], " ")
tf.print(c)
