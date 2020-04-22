
############################
# 加载CSV数据
############################

import functools
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("titanic_train.csv" ,TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("titanic_eval.csv" ,TEST_DATA_URL)

# print(train_file_path)
# /Users/yanmk/.keras/datasets/titanic_train.csv

# 自定义一个通用的预处理器来选出数值特行列，并将它们拼接
class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels




def get_dataset(file_path ,**kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs
    )
    return dataset


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key ,value.numpy()))


def pack(features, label):
    return tf.stack(list(features.values()),axis=-1), label


def normalize_numeric_data(data,mean,std):
    return (data - mean)/std




if __name__ == '__main__':
    ##############################################################
    ###### 1.加载数据
    LABEL_COLUMN = 'survived'
    LABELS = [0, 1]
    np.set_printoptions(precision=3,suppress=True) # 使numpy数值更易读，最多保留3位小数

    raw_train_data = get_dataset(train_file_path)
    raw_test_data = get_dataset(test_file_path)

    # show_batch(raw_train_data)

    # SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'class', 'deck', 'alone']
    # temp_dataset = get_dataset(train_file_path,select_columns=SELECT_COLUMNS)
    # # 可以自动选择出想要的列，通过设置select_columns实现；如果CSV文件中第一行不包含列名，可以自建列名list通过设置column_names实现
    # show_batch(temp_dataset)

    ##############################################################
    ###### 2.数据预处理
    SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
    DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
    temp_dataset = get_dataset(train_file_path,
                               select_columns=SELECT_COLUMNS,
                               column_defaults=DEFAULTS)

    # show_batch(temp_dataset)

    # example_batch, labels_batch = next(iter(temp_dataset))
    # print(example_batch)
    # print(labels_batch)

    # packed_dataset = temp_dataset.map(pack)
    # for features, labels in packed_dataset.take(1):
    #     print(features.numpy())
    #     print()
    #     print(labels.numpy())

    # 自定义一个通用的预处理器来选出数值特行列，并将它们拼接
    NUMERIC_FEATURES = ['age', 'n_siblings_spouses', 'parch', 'fare']
    packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
    packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
    # show_batch(packed_train_data)

    # 数据归一化
    desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
    # print(desc)

    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])


    #创建一个可用于***的normalizer
    normalizer = functools.partial(normalize_numeric_data,mean = MEAN, std = STD)

    numeric_column = tf.feature_column.numeric_column('numeric',normalizer_fn= normalizer,
                                                      shape=[len(NUMERIC_FEATURES)])

    numeric_columns = [numeric_column]

    # print(numeric_column)

    example_batch, labels_batch = next(iter(packed_train_data))

    numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
    # print(numeric_layer(example_batch).numpy())


    # 处理离散数据列
    CATEGORIES = {
        'sex': ['male', 'female'],
        'class': ['First', 'Second', 'Third'],
        'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
        'alone': ['y', 'n']
    }

    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    # pprint(categorical_columns)

    # categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
    # print(categorical_layer(example_batch).numpy()[0])

    preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)

    # print(preprocessing_layer(example_batch).numpy()[0])

    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])

    train_data = packed_train_data.shuffle(500)
    test_data = packed_test_data

    model.fit(train_data, epochs=20)

    test_loss, test_accuracy = model.evaluate(test_data)

    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

    predictions = model.predict(test_data)

    # Show some results
    for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
        prediction = tf.sigmoid(prediction).numpy()
        print("Predicted survival: {:.2%}".format(prediction[0]),
              " | Actual outcome: ",
              ("SURVIVED" if bool(survived) else "DIED"))
