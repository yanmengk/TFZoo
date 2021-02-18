import os

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

print(tf.__version__)

# 测试 加载图像文件数据集
# train_horse_dir = os.path.join("/Users/yanmk/programs/PythonProgram/TensorFlow/horse-or-human/horses")
# train_human_dir = os.path.join("/Users/yanmk/programs/PythonProgram/TensorFlow/horse-or-human/humans")

# print(os.listdir(train_horse_dir)[:10]) # ['horse43-5.png', 'horse06-5.png', 'horse20-6.png'...]
# print(os.listdir(train_human_dir)[:10]) # ['human17-22.png', 'human10-17.png', 'human10-03.png'...]

# 【数据部分】
## 数据预处理
train_datagen = ImageDataGenerator(rescale=1 / 255)
validation_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(
    directory='/Users/yanmk/programs/PythonProgram/TensorFlow/horse-or-human/',
    target_size=(300, 300),
    class_mode='binary',  # the order of the classes, which will map to the label indices, will be alphanumeric
    batch_size=32
)
validation_generator = validation_datagen.flow_from_directory(
    directory='/Users/yanmk/programs/PythonProgram/TensorFlow/validation-horse-or-human/',
    target_size=(300, 300),
    class_mode='binary',  # the order of the classes, which will map to the label indices, will be alphanumeric
    batch_size=32
)

# 【模型部分】
model = tf.keras.models.Sequential([
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# # model.summary()
# model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
#
# # 【模型训练部分】
# model.fit(train_generator,
#           steps_per_epoch=8,
#           epochs=15,
#           verbose=1,
#           validation_data=validation_generator,
#           validation_steps=8)
#
# # model.evaluate


# 【使用KerasTuner Hyperband进行网格化 超参数确定】
## 创建HyperParameters对象，然后在模型中插入Choice、Int等调参用的对象。
hp = HyperParameters()


def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(filters=hp.Choice('num_filters_top_layer', values=[16, 64], default=16),
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(300, 300, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for i in range(hp.Int("num_conv_layers", 1, 3)):
        model.add(Conv2D(hp.Choice(f'num_filters_layer{i}', values=[16, 64], default=16), (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(hp.Int("hidden_units", 128, 512, step=32), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
    return model


## 创建Hyperband对象
tuner = Hyperband(
    build_model,
    objective='val_acc',
    max_epochs=10,
    directory='horse_human_params',
    hyperparameters=hp,
    project_name='my_horse_human_project'
)

# tuner.search(train_generator, epochs=10, validation_data=validation_generator)

# best_hps = tuner.get_best_hyperparameters(1)[0]
# print(best_hps.values)
# model = tuner.hypermodel.build(best_hps)
# model.summary()
