import tensorflow as tf


# 自定义MyCallBack
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # if logs.get('loss') < 0.14:
        #     print("\n the loss is lower than 0.14, stop training!")
        #     self.model.stop_training = True

        if logs.get('accuracy') > 0.93:
            print("\n the accuracy is greater than 0.93, stop training!")
            self.model.stop_training = True


if __name__ == '__main__':
    print(tf.__version__)

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # filters=32, kernel_size=(3,3), 默认步长strides=(1, 1)
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # 默认池化size为pool_size=(2, 2) 会把输入张量的两个维度都缩小一半。
        # strides: 整数，或者2个整数表示的元组，或者是None。表示步长值。如果是 None，那么默认值是pool_size值
        # padding: "valid" 或者 "same" （区分大小写）。当是valid时不进行零填充，当是same时表示进行零填充，
        # 因此当strides=1时进行零填充后大小shape保持不变
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = myCallBack()

    model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    test_loss, test_auc = model.evaluate(x_test, y_test)
    print(test_auc)  # 0.9118
