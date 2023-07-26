import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
import os
import argparse
import logging
import logging.config
from tensorflow import keras
from model import LSTM_model, GRU_model, BiLSTM_model
from src.utils.metrics import calculate_loss
from src.utils.write_result import write_result
from src import config

# Model params
n_units = 150
dropout = .2
learning_rate = 1e-3
epochs = 200
batch_size = 256

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('loggers')

parser = argparse.ArgumentParser(
                    prog='WaterLevelPrediction',
                    description='Preprocessing data for the water prediction paper')

parser.add_argument('-p', '--pretrain')
parser.add_argument('-m', '--model')
parser.add_argument('-n', '--n_steps', type=int)
parser.add_argument('-l', '--lead_time', type=int)

args = parser.parse_args()

def train_model(trainX, trainY, testX, testY, trueY, model_name,
                pretrain=False, n_units=n_units, dropout=dropout,
                lr=learning_rate, epochs=epochs, batch_size=batch_size):

    if model_name == 'LSTM':
        model = LSTM_model(trainX, n_units)
    elif model_name == 'GRU':
        model = GRU_model(trainX, n_units)
    elif model_name == 'Bi-LSTM':
        model = BiLSTM_model(trainX, n_units)

    if pretrain:
        logger.info('Loading pretrain model...')
        model = keras.models.load(f'../../model/{model_name}_new.h5')
    else:
        logger.info('Training model...')
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        history = model.fit(trainX, trainY,
                            epochs=epochs,
                            batch_size=batch_size, validation_data=(testX, testY), verbose=0,
                            shuffle=False,
                            callbacks=[callback])

    predY = model.predict(testX)
    r2, rmse, mae, max_val_error = calculate_loss(trueY[args.n_steps+args.lead_time:], predY)
    logger.info('Metrics: ')
    logger.info(f'R^2: {r2}')
    logger.info(f'RMSE: {rmse}')
    logger.info(f'MAE: {mae}')
    logger.info(f'Max Error Value: {max_val_error}')
    logger.info('-'*30)
    model.save(f'../../model/{model_name}_new.h5')
    logger.info(f'Model saved at ../../model/{model_name}_new.h5')

def predict(model_weights_path):
    # Loads the weights
    model.load_weights(checkpoint_path)

if __name__ == '__main__':
    # Load data
    os.chdir(os.getcwd() + r'/data/postprocess')
    trainX = pickle.load(open('x_train_rescale.pkl', 'rb'))
    trainY = pickle.load(open('y_train_rescale.pkl', 'rb'))
    testX = pickle.load(open('x_test_rescale.pkl', 'rb'))
    testY = pickle.load(open('y_test_rescale.pkl', 'rb'))
    trueY = pickle.load(open('y_test.pkl', 'rb'))

    train_model(trainX, trainY, testX, testY, trueY,
                model_name=args.model,
                pretrain=args.pretrain)