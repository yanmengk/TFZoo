import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout

print(tf.__version__)
# 2.2.0

model = tf.keras.models.Sequential([
    Flatten(input_shape=(28, 28)),  # 或者 Flatten()
    Dense(128, activation='relu'),  # 或者 activation=tf.nn.relu
    Dropout(0.2),  # rate: Float between 0 and 1. Fraction of the input units to drop.
    Dense(10, activation='softmax')  # 或者 activation=tf.nn.softmax
])

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    # load_data()不指定特定位置，默认存储在~/.keras/datasets处。
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # 默认索引到"/Users/yanmk/.keras/datasets/mnist.npz"文件
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model.compile(optimizer=tf.optimizers.Adam(),  # 或者'adam'
                  loss=tf.losses.SparseCategoricalCrossentropy(),  # 或者'sparse_categorical_crossentropy'
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])  # 或者'accuracy'

    # 注意：The meaning of 'accuracy' depends on the loss function.
    # 当loss使用sparse_categorical_crossentropy的时候，应该使用tf.metrics.SparseCategoricalAccuracy()度量准确率,
    # 而非tf.metrics.Accuracy().

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)  # model在测试集上准确率为0.9774
