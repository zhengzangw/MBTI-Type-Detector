# Command line parameter
m: choose model  
s: sequence by sequence  
c: class(16/4)
e: enable early stop

# Avalible model
| name | description                   | accuracy |
| --   | ---                           | --       |
| cnn  | cnn+maxpool+cnn+maxpool+dense | 75%      |
| lstm | lstm+lstm+dense               |          |


# csv column
|name | description|
|-----|------------|
|type| 16 types|
|posts| long sentence, maxlength=2312|
|IE | I=1,E=0 |
|NS | N=1,S=0 |
|TF | T=1,F=0 |
|JP | J=1,P=0 |
___
# Models list
## CNN1

	model.add(keras.layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1))
   	model.add(keras.layers.MaxPool1D(2))
   	model.add(keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=2))
   	model.add(keras.layers.MaxPool1D(2))
   	model.add(keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=3))
   	model.add(keras.layers.GlobalMaxPool1D())
   	model.add(keras.layers.Dropout(0.5))
   	model.add(keras.layers.Dense(128, activation='relu'))

    [ 2019-01-16 11:22:45,203][end2end] Loss on test set(10%) = 0.6260359627847649
    [ 2019-01-16 11:22:45,415][end2end] Accuracy( for IE ) on test set(10%) = 0.7547605308713214
    [ 2019-01-16 11:22:45,415][end2end] [[285.0, 299.0], [126.0, 1023.0]]
    [ 2019-01-16 11:22:45,416][end2end] Accuracy( for NS ) on test set(10%) = 0.7882285054818234
    [ 2019-01-16 11:22:45,416][end2end] [[227.0, 348.0], [19.0, 1139.0]]
    [ 2019-01-16 11:22:45,416][end2end] Accuracy( for TF ) on test set(10%) = 0.6330063473744951
    [ 2019-01-16 11:22:45,416][end2end] [[424.0, 121.0], [515.0, 673.0]]
    [ 2019-01-16 11:22:45,416][end2end] Accuracy( for NP ) on test set(10%) = 0.7005193306405078
    [ 2019-01-16 11:22:45,416][end2end] [[799.0, 263.0], [256.0, 415.0]]
    [ 2019-01-16 11:22:45,416][end2end] Accuracy(Total) on test set(10%) = 0.33698788228505483
    [ 2019-01-16 11:22:45,416][end2end] Accuracy(One by one) on test set(10%) = 0.7191286785920369
    
## LSTM1
