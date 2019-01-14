import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

print(tf.__version__)

np.random.seed(1234)
tf.set_random_seed(1234)


# Input doc
df = pd.read_csv('MBTIv1.csv')
df = shuffle(df)

docs = df['posts']
labels = np.vstack([df['IE'],df['NS'],df['TF'],df['JP']]).transpose()

t = keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(docs)
max_length = 200
padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# Input gloVe
embeddings_index = {}
f = open('glove.6B.50d.txt','r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word]=coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 50))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Data splitting
num_instances = len(docs)
d1 = int(num_instances * 0.8)
d2 = int(num_instances * 0.9)

trainX = padded_docs[:d1]
trainY = labels[:d1, :]

valX = padded_docs[d1:d2]
valY = labels[d1:d2, :]

testX = padded_docs[d2:]
testY = labels[d2:, :]

# Define cnn model
model = keras.Sequential()
e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv1D(64, 3, padding='valid',activation='relu',strides=1))
model.add(keras.layers.GlobalMaxPool1D())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(4, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
history = model.fit(trainX, trainY, epochs=10, verbose=1, validation_data = (valX, valY))

# Testing
loss, accuracy = model.evaluate(testX, testY, verbose=1)

# Print the results
print("Loss on test set(10%) = {}".format(loss))
print("Accuracy on test set(10%) = {}".format(accuracy))
