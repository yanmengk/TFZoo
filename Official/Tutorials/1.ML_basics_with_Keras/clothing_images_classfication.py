import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Flatten, Dense

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Import the Fashion MNIST dataset
'''
训练集为 60,000 张 28x28 像素灰度图像，
测试集为 10,000 同规格图像，
总共 10 类时尚物品标签。
该数据集可以用作 MNIST 的直接替代品。类别标签是：

类别	描述	中文
0	T-shirt/top	T恤/上衣
1	Trouser	    裤子
2	Pullover	套头衫
3	Dress	    连衣裙
4	Coat	    外套
5	Sandal	    凉鞋
6	Shirt	    衬衫
7	Sneaker	    运动鞋
8	Bag	        背包
9	Ankle boot	短靴
'''

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)  # (60000, 28, 28)
print(len(train_labels))  # 60000
print(train_labels)  # [9 0 0 ... 3 0 5]
print(test_images.shape)  # (10000, 28, 28)
print(len(test_labels))  # 10000

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 数据预处理
## 将训练集和测试机的图片像素值Scale为0-1之间
train_images = train_images / 255.0
test_images = test_images / 255.0


# 建立模型
model = keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=10)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc)  # Test accuracy:  0.8821

predictions = model.predict(test_images)
print(predictions[0])
pred_label0 = np.argmax(predictions[0])
print("pred_label0: ", pred_label0)
true_label0 = test_labels[0]
print("true_label0: ", true_label0)
print(true_label0 == pred_label0)


