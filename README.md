# Introduction
This is ML model trained to detect people's MBTI type based what they write down in their post. There is also a demo that you can put in your daily life setence to detect which type you may be and see which word sells you out.
## MBTI dataset
kaggle dataset: https://www.kaggle.com/datasnaek/mbti-type/kernels 
The Myers-Briggs Type Indicator (MBTI) is a widely-used personality assessment tool. It has four pairs of preferences: 

* Introversion/Extraversion
* Intuition/Sensing
* Feeling/Thinking
* Perception/Judging


# Run Demo
    make demo
    
You need demo_cnn.h5, tokenizer.p to run the demo

# Training model
Add your model in models, then run

    python end2end.py -m your_model_name [-e|-s|-c]
    
You need MBTI.csv, glove.6B.50d.txt to train the demo
___
# Model Infomation
CNN info:
![Model](https://github.com/zhengzangw/ml-winter-camp/blob/master/pic/model.png)

    [ 2019-01-17 07:05:59,126][end2end]
    Accuracy(Total) on test set(10%) = 0.5362232779097387
    Accuracy(One by one) on test set(10%) = 0.813687648456057
    [I] precision: 0.544, recall: 0.919, f1: 0.684
    [E] precision: 0.970, recall: 0.774, f1: 0.861
    [N] precision: 0.509, recall: 0.987, f1: 0.672
    [S] precision: 0.998, recall: 0.853, f1: 0.920
    [T] precision: 0.890, recall: 0.763, f1: 0.821
    [P] precision: 0.767, recall: 0.892, f1: 0.825
    [J] precision: 0.808, recall: 0.787, f1: 0.797
    [F] precision: 0.673, recall: 0.701, f1: 0.686
