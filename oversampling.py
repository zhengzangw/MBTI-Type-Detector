import pandas as pd
import numpy as np
from sklearn.utils import shuffle

a = {
    'INFP': 1832,
    'INFJ': 1470,
    'INTP': 1304,
    'INTJ': 1091,
    'ENTP': 685,
    'ENFP': 675,
    'ISTP': 337,
    'ISFP': 271,
    'ENTJ': 231,
    'ISTJ': 205,
    'ENFJ': 190,
    'ISFJ': 166,
    'ESTP': 89,
    'ESFP': 48,
    'ESFJ': 42,
    'ESTJ': 39
}
weight = {
    'INFP': 1,
    'INFJ': 1,
    'INTP': 1,
    'INTJ': 2,
    'ENTP': 3,
    'ENFP': 3,
    'ISTP': 6,
    'ISFP': 7,
    'ENTJ': 8,
    'ISTJ': 9,
    'ENFJ': 9,
    'ISFJ': 10,
    'ESTP': 20,
    'ESFP': 38,
    'ESFJ': 39,
    'ESTJ': 40
}
per = ['I', 'E', 'N', 'S', 'T', 'F', 'J', 'P']
four_dim = ['IE', 'NS', 'TF', 'JP']
MBTI_label = [['E', 'I'],['S', 'N'],['F', 'T'],['P', 'J']]
num = {}

ORIGINAL_CSV = "MBTI.csv"

def adjust_weight():
    for t in per:
        num[t] = 0
    for key, value in a.items():
        for i in range(len(per)):
            if per[i] in key:
                num[per[i]] += value * weight[key]

def oversampling_csv(docs, labels, is_seq=False):
    msk = np.random.rand(len(docs)) < 0.8
    docs_train = docs[msk]
    label_train = labels[msk]
    testX = docs[~msk]
    testY = labels[~msk]

    trainX = None
    trainY = None
    u = list(weight.keys())
    for i in range(len(u)):
        msk = []
        for label in label_train:
            flag = True
            for k in range(4):
                if (MBTI_label[k][label[k]] not in u[i]):
                    flag = False
            msk.append(flag)

        inc_trainX = docs_train[msk].repeat(weight[u[i]], axis=0)
        inc_trainY = label_train[msk].repeat(weight[u[i]], axis=0)
        #print(u[i])
        #print("X={}, Y={}".format(len(inc_trainX), len(inc_trainY)))
        trainX = np.append(trainX, inc_trainX, axis=0) if trainX is not None else inc_trainX
        trainY = np.append(trainY, inc_trainY, axis=0) if trainY is not None else inc_trainY

    c = list(zip(trainX, trainY))
    shuffle(c)
    trainX[:], trainY[:] = zip(*c)

    return trainX, trainY, testX, testY
