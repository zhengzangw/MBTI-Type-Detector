import pandas as pd
from sklearn.utils import shuffle
import numpy as np

np.random.seed(1234)
import tensorflow as tf
from tensorflow import keras

tf.set_random_seed(1234)

from log_utils import get_logger
LOGGER = get_logger("end2end")

from models import get_model

MAX_LENGTH = 2300
VOCAB_SIZE = 0
MODEL_NAME = ""
CSV_NAME = "MBTIv1.csv"

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, help="Choose Your Model")
    parser.add_argument("--seq", dest="is_seq", action='store_true', help="Is test on sequence")
    parser.add_argument("--load", dest="loadpath", type=str, help="Load Model")
    parser.add_argument("--update_token", dest="utoken", action='store_true', help="Update Token or not")
    args = parser.parse_args()
    return args

def input_doc():
    global VOCAB_SIZE
    df = pd.read_csv(CSV_NAME)
    df = shuffle(df)

    docs = df['posts']
    labels = np.vstack([df['IE'], df['NS'], df['TF'], df['JP']]).transpose()

    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(docs)

    VOCAB_SIZE= len(t.word_index) + 1
    encoded_docs = t.texts_to_sequences(docs)
    padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=MAX_LENGTH, padding='post')
    return t,padded_docs,labels

def get_embedding_matrix(t):
    # Input gloVe
    embeddings_index = {}
    f = open('glove.6B.50d.txt', 'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    LOGGER.info('Loaded %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((VOCAB_SIZE, 50))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def data_splitting(docs,labels):
    num_instances = len(docs)
    d1 = int(num_instances * 0.8)
    d2 = int(num_instances * 0.9)

    trainX = docs[:d1]
    trainY = labels[:d1, :]

    valX = docs[d1:d2]
    valY = labels[d1:d2, :]

    testX = docs[d2:]
    testY = labels[d2:, :]
    return trainX, trainY, valX, valY, testX, testY

import pickle
def dump_tokenizer(t):
    pickle.dump(t, open("tokenizer.p", "wb"))

if __name__=="__main__":
    args = parse_args()
    MODEL_NAME = args.model
    CSV_NAME = "MBTIv2.csv" if args.is_seq else "MBTIv1.csv"
    MAX_LENGTH = 400 if args.is_seq else 2300

    t,padded_docs,labels = input_doc()
    embedding_matrix = get_embedding_matrix(t)
    dump_tokenizer(t)

    trainX,trainY,valX,valY,testX,testY = data_splitting(padded_docs,labels)

    if args.loadpath is None:
        model = get_model(MODEL_NAME, VOCAB_SIZE, embedding_matrix, MAX_LENGTH)
        model.summary(print_fn=LOGGER.info)

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                     keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Training
        LOGGER.info("Begin Training")
        history = model.fit(trainX, trainY, epochs=30, callbacks=callbacks, verbose=1,
                            validation_data=(valX, valY))

        # Testing
        loss, accuracy = model.evaluate(testX, testY, verbose=1)

        # Print the results
        LOGGER.info("\n------------------------\n")
        LOGGER.info("Loss on test set(10%) = {}".format(loss))
        LOGGER.info("Accuracy on test set(10%) = {}".format(accuracy))

        # Save the model
        model.save(MODEL_NAME+".h5")
    else:
        model = keras.models.load_model(args.loadpath)
        t, padded_docs, labels = input_doc()
        trainX, trainY, valX, valY, testX, testY = data_splitting(padded_docs, labels)
        loss, accuracy = model.evaluate(trainX, trainY, verbose=1)

        # Print the results
        LOGGER.info("\n------------------------\n")
        LOGGER.info("Loss on test set(10%) = {}".format(loss))
        LOGGER.info("Accuracy on test set(10%) = {}".format(accuracy))