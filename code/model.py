from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Input, Flatten, add
from tensorflow.keras.models import Model
import keras

def LSTM_model(X, n_units=150):
    model = Sequential()
    model.add(LSTM(n_units, kernel_initializer=keras.initializers.glorot_uniform(seed=seed), input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    return model

def GRU_model(X, n_units=150):
    model = Sequential()
    model.add(GRU(n_units, kernel_initializer=keras.initializers.glorot_uniform(seed=seed), input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    return model

def BiLSTM_model(X, n_units=150):
    model = Sequential()
    model.add(Bidirectional(LSTM(n_units, kernel_initializer=keras.initializers.glorot_uniform(seed=seed), input_shape=(X.shape[1], X.shape[2]))))
    model.add(Dense(1))
    return model
