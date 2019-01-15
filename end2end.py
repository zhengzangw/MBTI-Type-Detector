import pandas as pd
from sklearn.utils import shuffle
import numpy as np
np.random.seed(1234)
import tensorflow as tf
from tensorflow import keras
tf.set_random_seed(1234)

from log_utils import get_logger
LOGGER = get_logger("gcforest.gcforest")

from models import get_model

MAX_LENGTH = 2000

# Input doc
df = pd.read_csv('MBTIv1.csv')
df = shuffle(df)

docs = df['posts']
labels = np.vstack([df['IE'],df['NS'],df['TF'],df['JP']]).transpose()

t = keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(docs)

padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=MAX_LENGTH, padding='post')

# Input gloVe
embeddings_index = {}
f = open('glove.6B.50d.txt','r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word]=coefs
f.close()
LOGGER.info('Loaded %s word vectors.' % len(embeddings_index))

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

model = get_model("cnn_model",vocab_size,embedding_matrix,MAX_LENGTH)

model.summary(print_fn=LOGGER.info)
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
             keras.callbacks.ModelCheckpoint(filepath='best_model', monitor='val_loss', save_best_only=True)]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Training
LOGGER.info("Begin Training")
history = model.fit(trainX, trainY, epochs=100, callbacks=callbacks, verbose=1,
                    validation_data = (valX, valY))
LOGGER.info(history)

# Testing
loss, accuracy = model.evaluate(testX, testY, verbose=1)

# Print the results
LOGGER.info("Loss on test set(10%) = {}".format(loss))
LOGGER.info("Accuracy on test set(10%) = {}".format(accuracy))