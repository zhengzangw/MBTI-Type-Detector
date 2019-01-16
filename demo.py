import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from cleandata import deal_with_URL, deal_with_emoji

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", dest="loadpath", default="cnn16seq.h5", type=str, help="Load Model")
    args = parser.parse_args()
    return args

# 初始化
args = parse_args()
tf.set_random_seed(1234)
from log_utils import get_logger
LOGGER = get_logger("demos")
MAX_LENGTH = 2300
import pickle
MBTI_pos = ['I', 'N', 'T', 'J']
MBTI_neg = ['E', 'S', 'F', 'P']
model, tokenizer = None, None
# init end

def load_model_and_data(path):
    global model, tokenizer
    model = keras.models.load_model(path)
    tokenizer = pickle.load(open("tokenizer.p", "rb"))


def output_persenality(persenality, original=False):
    MBTI_tag = ""
    for i in range(4):
        MBTI_tag += MBTI_pos[i] if persenality[0][i] > 0.5 else MBTI_neg[i]
    if original:
        print("Your MBTI type maybe: {}".format(MBTI_tag))
        print("WHERE")
        for t in range(4):
            if persenality[0][t] > 0.5:
                print("\t{}:({:.1f})".format(MBTI_pos[t], persenality[0][t]))
            else:
                print("\t{}:({:.1f})".format(MBTI_neg[t], 1-persenality[0][t]))

def gen_color(sentence):
    df = pd.DataFrame(columns=['posts'])
    df.loc[0] = [sentence]
    deal_with_URL(df)
    deal_with_emoji(df)
    sentence = df['posts'][0]
    sentence = sentence.replace(',',' , ').replace('.', ' . ').replace('  ',' ')
    print('Processed Sentence: ', sentence)

    encoded_docs = tokenizer.texts_to_sequences([sentence])
    docs_len = len(encoded_docs[0])
    padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=MAX_LENGTH, padding='post')

    persenality = model.predict(padded_docs)
    output_persenality(persenality, original=True)


    modified_persenality_dict = {}
    for t in range(docs_len):
        temp = padded_docs[0][t]
        padded_docs[0][t] = 0
        modified_persenality = model.predict(padded_docs)
        output_persenality(modified_persenality)
        modified_persenality_dict[t] = modified_persenality
        padded_docs[0][t] = temp

    for t in range(docs_len):
        for i in range(4):
            modified_persenality_dict[t][0][i] -= persenality[0][i]
        # print(modified_persenality_dict[t][0])

    from sty import fg
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Color Control
    red = [100,0,0,0]
    green = [0,100,0,0]
    blue = [0,0,100,0]
    yellow = [0,0,0,100]
    red_inc = [1 if persenality[0][0]>0.5 else -1,0,0, 0]
    green_inc = [0,1 if persenality[0][1]>0.5 else -1,0,0]
    blue_inc = [0,0,1 if persenality[0][2]>0.5 else -1,0]
    yellow_inc = [0, 0, 0, 1 if persenality[0][2]>0.5 else -1]
    ans_dict = {'red': [], 'green': [], 'blue': [], 'yellow': []}
    cate = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        print("In {}th dimension, the sentense looks like:\n\t".format(i+1), end='')
        for t in range(docs_len):
            R = int(red[i] + 5 * modified_persenality_dict[t][0][i] * 1000 * red_inc[i])
            G = int(green[i] + 5 * modified_persenality_dict[t][0][i] * 1000 * green_inc[i])
            B = int(blue[i] + 5 * modified_persenality_dict[t][0][i] * 1000 * blue_inc[i])
            Y = int(yellow[i] + 5 * modified_persenality_dict[t][0][i] * 1000 * yellow_inc[i])
            G += int(float(Y)/2)
            R += int(float(Y)/2)
            ans_dict[cate[i]].append([R, G, B])
            print(fg(int(red[i]+5*modified_persenality_dict[t][0][i]*1000*red_inc[i]),
                     int(green[i]+5*modified_persenality_dict[t][0][i]*1000*green_inc[i]),
                     int(blue[i]+5*modified_persenality_dict[t][0][i]*1000*blue_inc[i]))
                  + reverse_word_map[padded_docs[0][t]] + fg.rs,end=' ')
        print()
    return ans_dict

if __name__=="__main__":
    load_model_and_data(args.loadpath)
    sentence = input('Please Input a sentence: ')

