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
## yeqy_CNN

    [ 2019-01-17 03:41:21,930][end2end] Accuracy(Total) on test set(10%) = 0.4570109636468552
    [ 2019-01-17 03:41:21,931][end2end] Accuracy(One by one) on test set(10%) = 0.765868436237738
    [I] precision: 0.445, recall: 0.869, f1: 0.588
    [E] precision: 0.942, recall: 0.663, f1: 0.778
    [N] precision: 0.472, recall: 0.935, f1: 0.628
    [S] precision: 0.987, recall: 0.827, f1: 0.900
    [T] precision: 0.842, recall: 0.705, f1: 0.768
    [P] precision: 0.707, recall: 0.844, f1: 0.770
    [J] precision: 0.784, recall: 0.792, f1: 0.788
    [F] precision: 0.672, recall: 0.661, f1: 0.666
    
## zzw_CNN

    [ 2019-01-17 03:33:00,094][end2end] Accuracy(Total) on test set(10%) = 0.42873629544143105
    [ 2019-01-17 03:33:00,094][end2end] Accuracy(One by one) on test set(10%) = 0.7478361223312175
    [I] precision: 0.437, recall: 0.820, f1: 0.570
    [E] precision: 0.923, recall: 0.671, f1: 0.777
    [N] precision: 0.456, recall: 0.947, f1: 0.616
    [S] precision: 0.989, recall: 0.813, f1: 0.893
    [T] precision: 0.828, recall: 0.669, f1: 0.740
    [P] precision: 0.681, recall: 0.836, f1: 0.751
    [J] precision: 0.771, recall: 0.738, f1: 0.754
    [F] precision: 0.618, recall: 0.659, f1: 0.638