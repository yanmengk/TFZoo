import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

tokenizer = Tokenizer()

data = open('/Users/yanmk/programs/PythonProgram/TensorFlow/sonnets.txt').read()
corpus = data.lower().split("\n")
print(corpus)

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # 加上一个 未登录词
# print(tokenizer.word_index)
# print(total_words)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# 对序列进行填充
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
# print(max_sequence_len) # 23

# 创建数据集和标签
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
# 对labels进行one_hot编码
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# 模型部分
model = tf.keras.models.Sequential([
    Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len - 1),
    Bidirectional(LSTM(units=150, return_sequences=True)),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words // 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(xs, ys, epochs=100)


def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


# plot_graph(history, 'accuracy')


# 预测文本内容
seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += ' ' + output_word

print(seed_text)
