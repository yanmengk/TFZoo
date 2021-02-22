import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do You think my dog is amazing?'
]

# tokenizer 分词器
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')  # num_words:需要保留的最大词数，基于词频。只有最常出现的 num_words-1 词会被保留。
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
# print(word_index) # {'my': 1, 'love': 2, 'dog': 3, 'i': 4, 'you': 5, 'cat': 6, 'do': 7, 'think': 8, 'is': 9, 'amazing': 10}

sequences = tokenizer.texts_to_sequences(sentences)
# print(sequences)


# test_data = [
#     'I really love my dog',
#     'my dog loves my manatee'
# ]
# print(tokenizer.texts_to_sequences(test_data)) # 不加oov_token的话：有些没有在训练集中的词被丢失，不会被编码 [[4, 2, 1, 3], [1, 3, 1]]


# 长度一致：加 pad_sequence
padded = pad_sequences(sequences, padding='post', maxlen=5, truncating='post')  # [8, 6, 9, 2, 4, 10, 11]]
# print(padded)
# [[ 5  3  2  4  0  0  0]
#  [ 5  3  2  7  0  0  0]
#  [ 6  3  2  4  0  0  0]
#  [ 8  6  9  2  4 10 11]]

# ******************处理真实的讽刺数据集实例*************
import json

sarcasm_data = '/Users/yanmk/programs/PythonProgram/TensorFlow/sarcasm.json'

with open(sarcasm_data, 'r') as f:
    datastore = json.load(f)

headlines = []
labels = []
urls = []
for data in datastore:
    headlines.append(data['headline'])
    labels.append(data['is_sarcastic'])
    urls.append(data['article_link'])

sarcasm_tokenizer = Tokenizer(oov_token='<OOV>')
sarcasm_tokenizer.fit_on_texts(headlines)

word_index = sarcasm_tokenizer.word_index
print(len(word_index))
# print(word_index)

sarcasm_sequences = sarcasm_tokenizer.texts_to_sequences(headlines)
sarcasm_padded = pad_sequences(sarcasm_sequences, padding='post')
print(headlines[2])
print(sarcasm_padded[2])
print(sarcasm_padded.shape)
