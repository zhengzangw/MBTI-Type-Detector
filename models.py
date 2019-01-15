from tensorflow import keras
from log_utils import get_logger
LOGGER = get_logger("models")

def get_model(name,vocab_size,embedding_matrix,input_length):
    if name=="cnn_model":
        return cnn_model(vocab_size,embedding_matrix,input_length)
    elif name=="new_cnn_model":
        return new_cnn_model(vocab_size,embedding_matrix,input_length)
    elif name=="lstm_model":
        return lstm_model(vocab_size,embedding_matrix,input_length)
    elif name=="zzw_model":
        return zzw_model(vocab_size, embedding_matrix, input_length)


def cnn_model(vocab_size,embedding_matrix,input_length):
    # maxlen=200: 72%
    # maxlen=2000: 75%
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(64, 3, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.GlobalMaxPool1D())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(4, activation='sigmoid'))
    return model

def new_cnn_model(vocab_size,embedding_matrix,input_length):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(100, 3, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPool1D())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(50, 10, padding='valid', activation='relu', strides=3))
    model.add(keras.layers.GlobalMaxPool1D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4, activation='sigmoid'))
    return model

def lstm_model(vocab_size,embedding_matrix,input_length):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.LSTM(20, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(16, 5, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.GlobalMaxPool1D())
    model.add(keras.layers.Dense(4, activation='sigmoid'))
    return model

def zzw_model(vocab_size,embedding_matrix,input_length):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv1D(64, 200, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv1D(64, 200, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(10, return_sequences=True))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4, activation='sigmoid'))
    return model