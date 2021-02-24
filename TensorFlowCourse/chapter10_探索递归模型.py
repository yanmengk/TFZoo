import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, GRU

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
# 为加快速度，仅使用少量数据
train_dataset, test_dataset = dataset['train'].take(4000), dataset['test'].take(1000)

# 利用数据集自带的分词器
tokenizer = info.features['text'].encoder

BUFFER_SIZE = 1000
BATCH_SIZE = 64
NUM_EPOCHS = 10

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_dataset.padded_batch(BATCH_SIZE)

model = tf.keras.Sequential([  # 等同于tf.keras.models.Sequential
    Embedding(tokenizer.vocab_size, 64),
    # Bidirectional(LSTM(64)),

    # Bidirectional(LSTM(8, return_sequences=True)),
    # Bidirectional(LSTM(8)),

    Bidirectional(GRU(32)),

    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')
