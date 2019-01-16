from tensorflow import keras
from log_utils import get_logger
LOGGER = get_logger("models")

def get_model(name,vocab_size,embedding_matrix,input_length, classify_type):
    if  name=="cnn":
        return cnn(vocab_size,embedding_matrix,input_length, classify_type)
    elif name=="lstm":
        return lstm(vocab_size,embedding_matrix,input_length, classify_type)
    else:
        LOGGER.error("no such model: {}".format(name))
        assert(0)


def final_active_func(classify_type):
    if classify_type == 4:
        return 'sigmoid'
    elif classify_type == 16:
        return 'softmax'

def cnn(vocab_size,embedding_matrix,input_length, classify_type):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPool1D(2))
    model.add(keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=2))
    model.add(keras.layers.MaxPool1D(2))
    model.add(keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=3))
    model.add(keras.layers.GlobalMaxPool1D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(classify_type, activation=final_active_func(classify_type)))
    return model

def lstm(vocab_size,embedding_matrix,input_length, classify_type):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.CuDNNLSTM(50, return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(50)))
    model.add(keras.layers.CuDNNLSTM(50, return_sequences=False))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2000))
    model.add(keras.layers.Dense(classify_type, activation=final_active_func(classify_type)))
    return model
