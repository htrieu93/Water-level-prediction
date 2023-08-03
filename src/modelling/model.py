import tensorflow as tf
import numpy as np
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Input, Flatten, add

# Reproducibility
seed = 42
os.PYTHONHASHSEED = 0
tf.keras.utils.set_random_seed(seed)
tf.config.experimental.enable_op_determinism()
np.random.seed(seed)
tf.random.set_seed(seed)

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
