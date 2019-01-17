# Command line parameter
m: choose model  
s: sequence by sequence  
c: class(16/4)
e: enable early stop

# Avalible model
| name | description                   | accuracy |
| --   | ---                           | --       |
| cnn  | cnn+maxpool+cnn+maxpool+dense |          |
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
## yeyq_CNN

	[ 2019-01-17 03:26:17,470][end2end] Loss on test set(10%) = 0.5259371216731888
    [ 2019-01-17 03:26:17,694][end2end] Accuracy( for IE ) on test set(10%) = 0.6912867859203693
    [ 2019-01-17 03:26:17,695][end2end] [[369.0, 493.0], [42.0, 829.0]]
    [ 2019-01-17 03:26:17,695][end2end] Accuracy( for NS ) on test set(10%) = 0.8349682631275245
    [ 2019-01-17 03:26:17,695][end2end] [[227.0, 267.0], [19.0, 1220.0]]
    [ 2019-01-17 03:26:17,695][end2end] Accuracy( for TF ) on test set(10%) = 0.7686093479515291
    [ 2019-01-17 03:26:17,695][end2end] [[638.0, 100.0], [301.0, 694.0]]
    [ 2019-01-17 03:26:17,695][end2end] Accuracy( for NP ) on test set(10%) = 0.7386035776110791
    [ 2019-01-17 03:26:17,695][end2end] [[835.0, 233.0], [220.0, 445.0]]
    [ 2019-01-17 03:26:17,695][end2end] Accuracy(Total) on test set(10%) = 0.4402769763416042
    [ 2019-01-17 03:26:17,695][end2end] Accuracy(One by one) on test set(10%) = 0.7583669936526255
    [I] precision: 0.428, recall: 0.898, f1: 0.580
    [E] precision: 0.952, recall: 0.627, f1: 0.756
    [N] precision: 0.460, recall: 0.923, f1: 0.614
    [S] precision: 0.985, recall: 0.820, f1: 0.895
    [T] precision: 0.864, recall: 0.679, f1: 0.761
    [P] precision: 0.697, recall: 0.874, f1: 0.776
    [J] precision: 0.782, recall: 0.791, f1: 0.787
    [F] precision: 0.669, recall: 0.656, f1: 0.663
    
## zzw_CNN

    model.add(keras.layers.Conv1D(128, 5, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPool1D(3))
    model.add(keras.layers.Conv1D(128, 5, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPool1D(5))
    model.add(keras.layers.Conv1D(128, 5, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPool1D(25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(classify_type, activation=final_active_func(classify_type)))
    model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

     2019-01-16 14:45:43,319][end2end] Loss on test set(10%) = 0.5272986194300501
    [ 2019-01-16 14:45:43,524][end2end] Accuracy( for IE ) on test set(10%) = 0.6162723600692441
    [ 2019-01-16 14:45:43,524][end2end] [[358.0, 612.0], [53.0, 710.0]]
    [ 2019-01-16 14:45:43,524][end2end] Accuracy( for NS ) on test set(10%) = 0.9128678592036931
    [ 2019-01-16 14:45:43,524][end2end] [[197.0, 102.0], [49.0, 1385.0]]
    [ 2019-01-16 14:45:43,524][end2end] Accuracy( for TF ) on test set(10%) = 0.7351413733410271
    [ 2019-01-16 14:45:43,524][end2end] [[748.0, 268.0], [191.0, 526.0]]
    [ 2019-01-16 14:45:43,524][end2end] Accuracy( for NP ) on test set(10%) = 0.6860934795152914
    [ 2019-01-16 14:45:43,524][end2end] [[747.0, 236.0], [308.0, 442.0]]
    [ 2019-01-16 14:45:43,524][end2end] Accuracy(Total) on test set(10%) = 0.363531448355453
    [ 2019-01-16 14:45:43,524][end2end] Accuracy(One by one) on test set(10%) = 0.7375937680323139
    [I] precision: 0.369, recall: 0.871, f1: 0.518
    [E] precision: 0.931, recall: 0.537, f1: 0.681
    [N] precision: 0.659, recall: 0.801, f1: 0.723
    [S] precision: 0.966, recall: 0.931, f1: 0.948
    [T] precision: 0.736, recall: 0.797, f1: 0.765
    [P] precision: 0.734, recall: 0.662, f1: 0.696
    [J] precision: 0.760, recall: 0.708, f1: 0.733
    [F] precision: 0.589, recall: 0.652, f1: 0.619