from tensorflow import keras
from log_utils import get_logger
LOGGER = get_logger("models")

def get_model(name,vocab_size,embedding_matrix,input_length):
    if name=="cnn":
        return cnn(vocab_size,embedding_matrix,input_length)
    elif name=="lstm":
        return lstm(vocab_size,embedding_matrix,input_length)
    else:
        LOGGER.error("no such model: {}".format(name))
        assert(0)

def cnn(vocab_size,embedding_matrix,input_length):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.Conv1D(64, 7, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPool1D(2))
    model.add(keras.layers.Conv1D(64, 7, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.GlobalMaxPool1D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(4, activation='sigmoid'))
    return model

def lstm(vocab_size,embedding_matrix,input_length):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.LSTM(50, return_sequences=True))
    model.add(keras.layers.LSTM(50, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(500)))
    model.add(keras.layers.Dense(4, activation='sigmoid'))
    return model
