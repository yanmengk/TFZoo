import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Flatten, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, RMSprop

print(tf.__version__)

sarcasm_data = '/Users/yanmk/programs/PythonProgram/TensorFlow/sarcasm.json'

# 参数设置
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
pad_type = 'post'
oov_tok = '<OOV>'
training_size = 20000
num_epochs = 30

with open(sarcasm_data, 'r') as f:
    datastore = json.load(f)

headlines = []
labels = []

for data in datastore:
    headlines.append(data['headline'])
    labels.append(data['is_sarcastic'])

train_data = headlines[:training_size]
train_label = labels[:training_size]
test_data = headlines[training_size:]
test_label = labels[training_size:]

sarcasm_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
sarcasm_tokenizer.fit_on_texts(train_data)

word_index = sarcasm_tokenizer.word_index
# print(len(word_index))
# print(word_index)

train_data = sarcasm_tokenizer.texts_to_sequences(train_data)
train_padded = pad_sequences(train_data, maxlen=max_length, padding=pad_type, truncating=trunc_type)
train_padded = np.array(train_padded)
train_label = np.array(train_label)

test_data = sarcasm_tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_data, maxlen=max_length, padding=pad_type, truncating=trunc_type)
test_padded = np.array(test_padded)
test_label = np.array(test_label)

model = tf.keras.models.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

history = model.fit(train_padded, train_label,
                    epochs=num_epochs,
                    validation_data=(test_padded, test_label),
                    verbose=2)  # verbose: 整数，0, 1 或 2。日志显示模式。 0=安静模式, 1=进度条, 2=每轮一行。默认为1


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
