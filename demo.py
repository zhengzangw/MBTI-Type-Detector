import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

tf.set_random_seed(1234)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", dest="loadpath", default="cnn_demo.h5", type=str, help="Load Model")
    args = parser.parse_args()
    return args

CSV_NAME = "MBTIv2.csv"
MAX_LENGTH = 400

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__=="__main__":
    args = parse_args()
    model = keras.models.load_model(args.loadpath)

    df = pd.read_csv(CSV_NAME)
    df = shuffle(df)
    docs = df['posts']
    labels = np.vstack([df['IE'], df['NS'], df['TF'], df['JP']]).transpose()
    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(docs)

    value = input('input a sentence: ')
    encoded_docs = t.texts_to_sequences([value])
    padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=MAX_LENGTH, padding='post')

    ans = model.predict(padded_docs)

    MBTI_pos = ['I', 'N', 'T', 'J']
    MBTI_neg = ['E', 'S', 'F', 'P']
    MBTI_tag = ""
    for i in range(4):
        MBTI_tag += MBTI_pos[i] if ans[0][i]>0.5 else MBTI_neg[i]

    print("Your MBTI type maybe: {}".format(MBTI_tag))
