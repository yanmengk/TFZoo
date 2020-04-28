import tensorflow as tf
from tf_wiki.basic.ex04_MLP_mnist import MNISTLoader


'''
与前面介绍的 Checkpoint 不同，SavedModel 包含了一个 TensorFlow 程序的完整信息： 不仅包含参数的权值，还包含计算的流程（即计算图） 。
当模型导出为 SavedModel 文件时，无需建立模型的源代码即可再次运行模型，这使得 SavedModel 尤其适用于模型的分享和部署。
'''

'''
Keras 模型均可方便地导出为 SavedModel 格式。不过需要注意的是，因为 SavedModel 基于计算图，
所以对于使用继承 tf.keras.Model 类建立的 Keras 模型，其需要导出到 SavedModel 格式的方法（比如 call ）都需要使用 @tf.function 修饰
'''

num_epochs =1
batch_size = 50
learning_rate = 0.001

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

data_loader = MNISTLoader()
# model.compile(
#     optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate),
#     loss=tf.keras.losses.sparse_categorical_crossentropy,
#     metrics=[tf.keras.metrics.sparse_categorical_accuracy]
# )
#
# # 导出模型文件
# model.fit(data_loader.train_data, data_loader.train_label,epochs=num_epochs,batch_size=batch_size)
# tf.saved_model.save(model,"saved/1")


# 从文件加载模型
model = tf.saved_model.load("saved/1")
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())
# test accuracy: 0.953000