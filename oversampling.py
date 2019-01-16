import pandas as pd

if __name__=="__main__":
    temp = pd.read_csv("MBTIv0.csv")
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
    num = {}
    for t in per:
        num[t] = 0
    for key,value in a.items():
        for i in range(len(per)):
            if per[i] in key:
                num[per[i]] += value*weight[key]

    oversample = pd.DataFrame(columns=temp.columns)
    u = list(weight.keys())
    for i in range(len(u)):
        newd = temp[temp["type"]==u[i]]
        cpy = ([newd]*weight[u[i]])
        cpy.append(oversample)
        oversample = pd.concat(cpy, ignore_index=True)
    oversample.to_csv('MBTIv0_oversample.csv')