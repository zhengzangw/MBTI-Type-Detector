import pandas as pd
from sklearn.utils import shuffle
import numpy as np

np.random.seed(1234)
import tensorflow as tf
from tensorflow import keras

tf.set_random_seed(1234)

from util_tools.log_utils import get_logger
LOGGER = get_logger("end2end")

from models import get_model
from util_tools.cleandata import split_sentence, get_the_label

# Glabol Variable
MAX_LENGTH = 0
VOCAB_SIZE = 0
MODEL_NAME = ""
CSV_NAME = "MBTI.csv"
CTYPE = 4
IS_SEQ = False
POS_CATEGORY = ['I', 'N', 'T', 'J']
NEG_CATEGORY = ['E', 'S', 'F', 'P']

# Parse args
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", type=str, default="zzw_lstm", help="Choose Your Model")
    parser.add_argument("-s", "--seq", dest="is_seq", action='store_true', help="Is test on sequence")
    parser.add_argument("-l", "--load", dest="loadpath", type=str, default=None, help="Load Model")
    parser.add_argument("-c", "--classify", dest="classify", type=int, default=4, help="Choose The Classify Method, 4/16")
    parser.add_argument("-e", "--early", dest="is_early_stop", action='store_true', help="Open early stop")
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
    if IS_SEQ:
        df = split_sentence(df)
        get_the_label(df)

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
    # print(np.mean([len(x) for x in encoded_docs]))
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

    embedding_matrix = np.random.normal(0,1,(VOCAB_SIZE, 50))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Split Data
from util_tools.oversampling import oversampling_csv
def data_splitting(docs, labels):
    global IS_SEQ
    trainX, trainY, valX, valY, testX, testY = oversampling_csv(docs, labels, IS_SEQ)
    return trainX, trainY, valX, valY, testX, testY


def testing(model, testX, testY):
    LOGGER.info("\n------------------------\n")
    loss, _ = model.evaluate(testX, testY, verbose=1) # 不用它的accuracy
    LOGGER.info("Loss on test set(10%) = {}".format(loss))

    # accuary
    counter_sep = [0, 0, 0, 0]
    counter_one_by_one = 0
    counter_total = 0
    X = model.predict(testX).tolist()
    Y = testY.tolist()
    shape = testY.shape

    if CTYPE == 4:
        confusion = np.zeros((4,2,2))
        for i in range(shape[0]):
            rowx = []
            rowy = []
            for j in range(shape[1]):
                bx = X[i][j] > 0.5
                by = Y[i][j] > 0.5
                counter_one_by_one += int(bx == by)
                counter_sep[j] += int(bx == by)
                confusion[j][int(bx)][int(by)] += 1
                rowx.append(bx)
                rowy.append(by)
            counter_total += int(rowx == rowy)
        cate = ['IE', 'NS', 'TF', 'NP']
        for i in range(4):
            LOGGER.info("Accuracy( for {} ) on test set(10%) = {}".format(cate[i], float(counter_sep[i])/float(shape[0])))
            LOGGER.info(confusion[i].tolist())
        LOGGER.info("Accuracy(Total) on test set(10%) = {}".format(float(counter_total)/float(shape[0])))
        LOGGER.info("Accuracy(One by one) on test set(10%) = {}".format((float(counter_one_by_one)/float(shape[0] * 4))))
        print_pr(confusion)
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

def calc_pr(predicted, ground_truth):
    cnt = np.zeros((2,2))
    for i in range(len(predicted)):
        cnt[predicted[i]][ground_truth[i]] += 1
    return cnt[0][0]/(cnt[0][0]+cnt[0][1]), cnt[0][0]/(cnt[0][0]+cnt[1][0]),\
           cnt[1][1]/(cnt[1][1]+cnt[1][0]), cnt[1][1]/(cnt[1][1]+cnt[0][1])

def print_pr(confusion):
    for i in range(4):
        t = confusion[i]
        p = t[0][0]/(t[0][0]+t[0][1]+1e-10)
        r = t[0][0]/(t[0][0]+t[1][0]+1e-10)
        print('[%s] precision: %.3f, recall: %.3f, f1: %.3f' % \
              (POS_CATEGORY[i], p, r, 2.0/(1/p+1/r)))
        p = t[1][1]/(t[1][1]+t[1][0]+1e-10)
        r = t[1][1]/(t[1][1]+t[0][1]+1e-10)
        print('[%s] precision: %.3f, recall: %.3f, f1: %.3f' % \
              (NEG_CATEGORY[i], p, r, 2.0/(1/p+1/r)))


def plot_pr_roc(model, testX, testY):
    X = model.predict(testX)
    Y = testY
    predicted = X[:, 3]
    ground_truth = Y[:, 3]
    #print(predicted)
    #print(ground_truth)
    threshold_list = [1.0 * i / 100 for i in range(1, 100)]
    precision = []
    recall = []
    pre1 = []
    rec1 = []
    for thres in threshold_list:
        predicted_with_thres = [int(item > thres) for item in predicted]
        p, r, p1, r1 = calc_pr(predicted_with_thres, ground_truth)
        precision.append(p)
        recall.append(r)
        pre1.append(p1)
        rec1.append(r1)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 5))
    ax1.plot(recall, precision)
    ax2.plot(rec1, pre1)
    ax1.axis([0, 1, 0, 1])
    ax2.axis([0,1,0,1])
    plt.show()
    #fig = plt.figure(fig)

if __name__=="__main__":
    args = parse_args()
    MODEL_NAME = args.model
    IS_SEQ = args.is_seq
    MAX_LENGTH = 400 if IS_SEQ else 2300
    CTYPE = args.classify

    # Load Data
    docs, labels = input_doc()
    tokenizer = get_tokenizer(docs)
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    padded_docs = transfrom_doc(docs, tokenizer)
    embedding_matrix = get_embedding_matrix(tokenizer)

    trainX,trainY,valX,valY,testX,testY = data_splitting(padded_docs, labels)

    if args.loadpath is None:

        # Choose Evaluating Function
        if CTYPE != 4 and CTYPE != 16:
            assert (0)
        loss_func = 'binary_crossentropy' if CTYPE == 4 else 'categorical_crossentropy'

        if args.is_early_stop:
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=4),
                     keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        else:
            callbacks = None

        BATCH_SIZE = 128
        model = get_model(MODEL_NAME, VOCAB_SIZE, embedding_matrix, MAX_LENGTH, CTYPE, loss_func, BATCH_SIZE)
        model.summary(print_fn=LOGGER.info)

        # Training
        LOGGER.info("Begin Training")
        history = model.fit(trainX, trainY, epochs=80, verbose=1, callbacks=callbacks,
                            validation_data=(valX, valY), batch_size=BATCH_SIZE)

        # Testing
        LOGGER.info("Begin Testing")
        testing(model, testX, testY)

        # Save the model
        SAVE_NAME = MODEL_NAME+str(CTYPE)
        if args.is_seq:
            SAVE_NAME += "seq"
        model.save(SAVE_NAME+".h5")
    else:
        model = keras.models.load_model(args.loadpath)
        testing(model, testX, testY)
        # plot_pr_roc(model, testX, testY)
