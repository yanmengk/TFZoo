import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# plt.imshow(x_train[0])
# plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),  # activation = 'relu'
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),  # activation = 'softmax'
])

# print(model.summary())
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)

# preds = model.predict(x_test)
# print(np.argmax(preds[0]))
# print(y_test[0])

# 自定义callback，epoch达到某个准确率就停止训练
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy')) > 0.88:
            print("\nReached 99% accuracy, stop training model")
            self.model.stop_training = True


callbacks = myCallback()
model.fit(x_train, y_train, epochs=5, callbacks=[callbacks])
