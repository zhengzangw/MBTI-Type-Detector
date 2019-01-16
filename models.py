from tensorflow import keras
from log_utils import get_logger
LOGGER = get_logger("models")

sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam()

def get_model(name,vocab_size,embedding_matrix,input_length, classify_type, loss_function, batch_size):
    if  name=="zzw_cnn":
        return zzw_cnn(vocab_size,embedding_matrix,input_length, classify_type, loss_function, batch_size)
    elif name=="zzw_lstm":
        return zzw_lstm(vocab_size,embedding_matrix,input_length, classify_type, loss_function, batch_size)
    else:
        LOGGER.error("no such model: {}".format(name))
        assert(0)


def final_active_func(classify_type):
    if classify_type == 4:
        return 'sigmoid'
    elif classify_type == 16:
        return 'softmax'

def zzw_cnn(vocab_size,embedding_matrix,input_length, classify_type, loss_function, batch_size):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.Conv1D(128, 5, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPool1D(5))
    model.add(keras.layers.Conv1D(128, 5, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPool1D(5))
    model.add(keras.layers.Conv1D(128, 5, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPool1D(35))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(classify_type, activation=final_active_func(classify_type)))
    model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
    return model

def zzw_lstm(vocab_size,embedding_matrix,input_length, classify_type, loss_function, batch_size):
    model = keras.Sequential()
    e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(keras.layers.CuDNNLSTM(50, return_sequences=True))
    model.add(keras.layers.CuDNNLSTM(50, return_sequences=True))
    model.add(keras.layers.CuDNNLSTM(50, return_sequences=False))
    model.add(keras.layers.Dense(classify_type, activation=final_active_func(classify_type)))
    model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
    return model

