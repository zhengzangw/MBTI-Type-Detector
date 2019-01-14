# Preworks lib
import pandas as pd
import numpy as np
import re

# read from file
def input_csv(file_name):
    df = pd.read_csv(file_name)
    # Get rid of first and last character
    for t in range(len(df)):
        df['posts'][t] = df['posts'][t][1:-1]
    df.head()
    return df


# Get Label for MBTI
def get_the_label(df):
    MBTI =['IE', 'NS', 'TF', 'JP']
    MBTI_pos = ['I', 'N', 'T', 'J']
    MBTI_neg = ['E', 'S', 'F', 'P']
    for t in range(len(MBTI)):
        df[MBTI[t]] = [1 if df['type'][i][t]==MBTI_pos[t] else 0 for i in range(len(df))]


# Replace Return
def deal_with_seperator(df):
    RETURN_REGULAR_PATTERN = "\|\|\|"
    df['posts'] = df['posts'].str.replace(RETURN_REGULAR_PATTERN, " <RETURN> ", regex=True)

# Replace URL
def deal_with_URL(df):
    URL_REGULAR_PATTERN = "(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"
    df['posts'] = df['posts'].str.replace(URL_REGULAR_PATTERN, " <URL> ", regex=True)


# Get Label for emoji
def get_emoji_dict():
    f = open("./emoji_acsii.txt", "r")
    lines = f.read()
    lines = lines.split('\n')
    dict = []
    for line in lines:
        words = line.split(' ')
        tag = words[0]
        tag = tag.replace('-', ' ')
        words = words[1:-1]
        for word in words:
            dict.append([word, tag])
    f.close()
    return dict

# replace the emoji with the meanful words
def deal_with_emoji(df):
    emoji = get_emoji_dict()
    for pattern in emoji:
        df['posts'] = df['posts'].str.replace(pattern[0], pattern[1], regex=False)

# Split sentence by the symbol |||
def split_sentence(df):
    dict = {'type':[], 'posts':[]}
    l = len(df)
    for i in range(l):
        items = df['posts'][i].split(r'<RETURN>')
        for item in items:
            dict['type'].append(df['type'][i])
            dict['posts'].append(item)
    return pd.DataFrame(dict)

# Output the df file
def output_csv(df, file_name):
    df.to_csv(file_name)

# Main prework process for the whole dataset
def prework(input_file, output_file):
    df = input_csv(input_file)
    get_the_label(df)
    deal_with_seperator(df)
    deal_with_URL(df)
    deal_with_emoji(df)
    # selectable
    df = split_sentence(df)
    output_csv(df, output_file)

# Save df
if __name__ == '__main__':
    prework('./MBTIv0.csv', './MBTIv2.csv')
