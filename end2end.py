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

# Glabol Variable
MAX_LENGTH = 0
VOCAB_SIZE = 0
MODEL_NAME = ""
CSV_NAME = ""
CTYPE = 4

# Parse args
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", type=str, default="cnn", help="Choose Your Model")
    parser.add_argument("-s", "--seq", dest="is_seq", action='store_true', help="Is test on sequence")
    parser.add_argument("-l", "--load", dest="loadpath", type=str, default=None, help="Load Model")
    parser.add_argument("-c", "--classify", dest="classify", type=int, default=4, help="Choose The Classify Method, 4/16")
    args = parser.parse_args()
    return args

# Save Tokenizer
import pickle
def dump_tokenizer(t):
    pickle.dump(t, open("tokenizer.p", "wb"))

# Input CSV
def input_doc():
    df = pd.read_csv(CSV_NAME)
    df = shuffle(df)

    docs = df['posts']
    labels = np.vstack([df['IE'], df['NS'], df['TF'], df['JP']]).transpose()
    if CTYPE == 16:
        labels = labels[:, 0] * 8 + labels[:, 1] * 4 + labels[:, 2] * 2 + labels[:, 3]
        tmp = np.zeros([labels.shape[0], 16])
        for i in range(labels.shape[0]):
            tmp[i, labels[i]] = 1
        labels = tmp
        
    return docs, labels
    
# Get tokenizer
def get_tokenizer(docs):
    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(docs)
    dump_tokenizer(t)
    return t

# Encode and Pad docs
def transfrom_doc(docs,tokenizer):
    encoded_docs = tokenizer.texts_to_sequences(docs)
    padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=MAX_LENGTH, padding='post')
    return padded_docs

# Get doc2vec from gloVe
def get_embedding_matrix(t):
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

# Split Data
def data_splitting(docs, labels):
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


def testing(model, testX, testY):
    LOGGER.info("\n------------------------\n")
    loss, _ = model.evaluate(testX, testY, verbose=1) # 不用它的accuracy
    LOGGER.info("Loss on test set(10%) = {}".format(loss))

    # 计算accuary
    shape = testX.shape
    counter_sep = [0, 0, 0, 0]
    counter_one_by_one = 0
    counter_total = 0
    X = model.predict(testX).tolist()
    Y = testY.tolist()
    shape = testY.shape
    print("hhh{}".format(shape))

    if CTYPE == 4:
        for i in range(shape[0]):
            rowx = []
            rowy = []
            for j in range(shape[1]):
                bx = X[i][j] > 0.5
                by = Y[i][j] > 0.5
                counter_one_by_one += int(bx == by)
                counter_sep[j] += int(bx == by)
                rowx.append(bx)
                rowy.append(by)
            counter_total += int(rowx == rowy)
        cate = ['IE', 'NS', 'TF', 'NP']
        for i in range(4):
            LOGGER.info("Accuracy( for {} ) on test set(10%) = {}".format(cate[i], float(counter_sep[i])/float(shape[0])))
        LOGGER.info("Accuracy(Total) on test set(10%) = {}".format(float(counter_total)/float(shape[0])))
        LOGGER.info("Accuracy(One by one) on test set(10%) = {}".format((float(counter_one_by_one)/float(shape[0] * 4))))
    elif CTYPE == 16:
        for i in range(shape[0]):
            val, whex, whey = -1e20, -1, -1
            for j in range(shape[1]):
                if val < X[i][j]:
                    val, whex = X[i][j], j
            val = -1e20
            for j in range(shape[1]):
                if val < Y[i][j]:
                    val, whey = Y[i][j], j
            counter_total += int(whex == whey)
        LOGGER.info("Accuracy(Total) on test set(10%) = {}".format(float(counter_total)/float(shape[0])))


if __name__=="__main__":
    args = parse_args()
    MODEL_NAME = args.model
    CSV_NAME = "MBTIv2.csv" if args.is_seq else "MBTIv1.csv"
    MAX_LENGTH = 400 if args.is_seq else 2300
    CTYPE = args.classify

    docs, labels = input_doc()
    tokenizer = get_tokenizer(docs)
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    padded_docs = transfrom_doc(docs, tokenizer)
    embedding_matrix = get_embedding_matrix(tokenizer)

    trainX,trainY,valX,valY,testX,testY = data_splitting(padded_docs, labels)

    if args.loadpath is None:
        model = get_model(MODEL_NAME, VOCAB_SIZE, embedding_matrix, MAX_LENGTH, CTYPE)
        model.summary(print_fn=LOGGER.info)

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
                 keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    
        # 根据分类的函方式选择对应的函数
        if CTYPE!=4 and CTYPE!=16:
            assert(0)
        loss_func = 'binary_crossentropy' if CTYPE==4 else 'categorical_crossentropy'
        model.compile(loss = loss_func, optimizer='adam', metrics=['accuracy'])

        # Training
        LOGGER.info("Begin Training")
        history = model.fit(trainX, trainY, epochs=20, callbacks=callbacks, verbose=1,
                            validation_data=(valX, valY))

        # Testing
        testing(model, testX, testY)

        # Save the model
        model.save(MODEL_NAME+".h5")
    else:
        model = keras.models.load_model(args.loadpath)
        testing(model, testX, testY)