from tensorflow import keras
from log_utils import get_logger
LOGGER = get_logger("models")

def get_model(name,vocab_size,embedding_matrix,input_length, classify_type):
    if name=="demo_cnn":
        return demo_cnn(vocab_size,embedding_matrix,input_length)
    elif name=="two_level_cnn":
        return two_level_cnn(vocab_size,embedding_matrix,input_length)
    elif name=="two_level_lstm":
        return two_level_lstm(vocab_size,embedding_matrix,input_length)
    else:
        LOGGER.error("no such model: {}".format(name))


def demo_cnn(vocab_size,embedding_matrix,input_length, classify_type):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(64, 3, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.GlobalMaxPool1D())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(classify_type, activation='sigmoid'))
    return model
# maxlen=200: 72%
# maxlen=2000: 75%

def two_level_cnn(vocab_size,embedding_matrix,input_length, classify_type):
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
    model.add(keras.layers.Dense(classify_type, activation='sigmoid'))
    return model

def two_level_lstm(vocab_size,embedding_matrix,input_length, classify_type):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.LSTM(50, return_sequences=True))
    model.add(keras.layers.LSTM(50, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(vocab_size/10)))
    model.add(keras.layers.Dense(classify_type, activation='sigmoid'))
    return model
