import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from cleandata import deal_with_URL, deal_with_emoji
import PIL
from scipy.misc import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", dest="loadpath", default="demo_cnn.h5", type=str, help="Load Model")
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
DOCS_LEN = 0
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

def gen_color(sentence, for_wordcloud = False):
    df = pd.DataFrame(columns=['posts'])
    df.loc[0] = [sentence]
    deal_with_URL(df)
    deal_with_emoji(df)
    sentence = df['posts'][0]
    sentence = sentence.replace(',',' , ').replace('.', ' . ').replace('  ',' ')
    print('Processed Sentence: ', sentence)

    encoded_docs = tokenizer.texts_to_sequences([sentence])
    docs_len = len(encoded_docs[0])
    global DOCS_LEN
    DOCS_LEN = docs_len
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

    # 只需要计算云图的情况
    if for_wordcloud == True:
        return persenality, reverse_word_map, padded_docs, modified_persenality_dict

    # Color Control
    red = [100,0,0,0]
    green = [0,100,0,0]
    blue = [0,0,100,0]
    yellow = [0,0,0,100]
    red_inc = [1 if persenality[0][0]>0.5 else -1,0,0, 0]
    green_inc = [0,1 if persenality[0][1]>0.5 else -1,0,0]
    blue_inc = [0,0,1 if persenality[0][2]>0.5 else -1,0]
    yellow_inc = [0, 0, 0, 1 if persenality[0][2]>0.5 else -1]
    ans_dict = {0: [], 1: [], 2: [], 3: []}
    for i in range(4):
        print("In {}th dimension, the sentense looks like:\n\t".format(i+1), end='')
        for t in range(docs_len):
            R = int(red[i] + 5 * modified_persenality_dict[t][0][i] * 1000 * red_inc[i])
            G = int(green[i] + 5 * modified_persenality_dict[t][0][i] * 1000 * green_inc[i])
            B = int(blue[i] + 5 * modified_persenality_dict[t][0][i] * 1000 * blue_inc[i])
            Y = int(yellow[i] + 5 * modified_persenality_dict[t][0][i] * 1000 * yellow_inc[i])
            G += int(float(Y)/2)
            R += int(float(Y)/2)
            ans_dict[i].append([reverse_word_map[padded_docs[0][t]], [R, G, B]])
            print(fg(int(red[i]+5*modified_persenality_dict[t][0][i]*1000*red_inc[i]),
                     int(green[i]+5*modified_persenality_dict[t][0][i]*1000*green_inc[i]),
                     int(blue[i]+5*modified_persenality_dict[t][0][i]*1000*blue_inc[i]))
                  + reverse_word_map[padded_docs[0][t]] + fg.rs,end=' ')
        print()
    persenality_ret = []
    for i in range(4):
        persenality_ret.append(0 if persenality[0][i] > 0.5 else 1)
    return ans_dict, persenality_ret

def get_wordcloud(sentence):
    persenality, reverse_word_map, padded_docs, modified_persenality_dict = gen_color(sentence, True)
    fig, ax = plt.subplots(2, 2, figsize=(25, 7 * 2))

    for i in range(2):
        for j in range(2):
            temp = {}
            sgn = 1 if persenality[0][i] > 0.5 else -1
            for t in range(DOCS_LEN):
                temp[reverse_word_map[padded_docs[0][t]]] = int(1000 * modified_persenality_dict[t][0][i] * sgn)
            low_bound = -min(temp.values()) + 1
            for key in temp:
                temp[key] += low_bound
            #  print(temp)

            wc = WordCloud().generate_from_frequencies(temp)
            ax[i][j].imshow(wc)
            ax[i][j].set_title(MBTI_pos[2*i+j] if persenality[0][2]>0.5 else MBTI_neg[2*i+j], fontsize=50)
            ax[i][j].axis("off")

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    return pil_image

if __name__=="__main__":
    load_model_and_data(args.loadpath)
    sentence = input('Please Input a sentence: ')
    gen_color(sentence)

