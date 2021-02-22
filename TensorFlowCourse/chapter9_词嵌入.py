import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import io

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'

for s, l in train_data:
    train_sentences.append(str(s.numpy()))  # 此处需要将bytes类型转化为str类型
    train_labels.append(l.numpy())

for s, l in test_data:
    test_sentences.append(str(s.numpy()))
    test_labels.append(l.numpy())

train_labels_final = np.array(train_labels)
test_labels_final = np.array(test_labels)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
# print(word_index)

sequences = tokenizer.texts_to_sequences(train_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding='post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, truncating=trunc_type, padding='post')

reverse_word_index = dict([(value, key) for key, value in word_index.items()])

# def decode_review(text):
#     return ' '.join([reverse_word_index.get(i,'?') for i in text])
#
# print(reverse_word_index)
# print(decode_review(padded[1]))
# print(train_sentences[1])


model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),  # 也可以采用 tf.keras.layers.GlobalAveragePooling1D(),全局平均池化层
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

num_epochs = 10
model.fit(padded, train_labels_final, epochs=num_epochs, validation_data=(test_padded, test_labels_final))

# shape: (vocab_size, embedding_dim)
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)

# 将词向量写入到文件中
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embedding = weights[word_num]
    out_v.write('\t'.join([str(x) for x in embedding]) + '\n')
    out_m.write(word + '\n')

out_m.close()
out_v.close()
