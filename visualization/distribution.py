import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pandas as pd
import numpy as np


file = '../MBTIv1.csv'

df = pd.read_csv(file)

p = np.array([sum(df['IE']), sum(df['NS']), sum(df['TF']), sum(df['JP'])])
p = p/df.shape[0]
pos = ['I', 'N', 'T', 'J']
neg = ['E', 'S', 'F', 'P']

x = np.arange(4) * 0.8
fig = plt.figure(figsize=(5,5))
plt.bar(x, [1,1,1,1], width=0.4, align='center')
plt.bar(x, p, width=0.4, align='center')
plt.xticks(x, ['' for i in range(4)])
#plt.xticks(x, ['I vs. E', 'N vs. S', 'T vs. F', 'J vs. P'])
plt.xlabel('Personality Axes')
plt.ylabel('Proportion')
plt.title('Label Distribution of MBTI Dataset')
for i in range(4):
    plt.text(x[i], 0.1, pos[i], ha='center', color='w', size=15)
    plt.text(x[i], 0.9, neg[i], ha='center', color='w', size=15)
plt.savefig('./distribution.png')
plt.show()