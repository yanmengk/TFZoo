import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

print(tf.__version__)

model = tf.keras.models.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    # load_data()不指定特定位置，默认存储在~/.keras/datasets处。
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)  # model在测试集上准确率为0.9774
