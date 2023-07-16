import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
import os
import csv
import argparse
import logging
import logging.config
from model import LSTM_model, GRU_model, BiLSTM_model
from util import *
from config import *

# Model params
n_units = 100
dropout = .2
learning_rate = 1e-3
epochs = 200
batchsize = 256

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('loggers')

parser = argparse.ArgumentParser(
                    prog='WaterLevelPrediction',
                    description='Preprocessing data for the water prediction paper')

parser.add_argument('-p', '--pretrain')
parser.add_argument('-m', '--model')

args = parser.parse_args()

def train_model(trainX, trainY, testX, testY, trueY, model_name, pretrain=False):

    if model_name == 'LSTM':
        model = LSTM_model(trainX)
    elif model_name == 'GRU':
        model = GRU_model(trainX)
    elif model_name == 'Bi-LSTM':
        model = BiLSTM_model(trainX)

    # if pretrain:
    #    model =

    logger.info('Training model...')
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(trainX, trainY,
                        epochs=epochs,
                        batch_size=batchsize, validation_data=(testX, testY), verbose=0,
                        shuffle=False,
                        callbacks=[callback])

    predY = model.predict(testX)
    r2, rmse, mae, max_val_error = calculate_loss(trueY[n_steps+lead_time:], predY)
    logger.info('Metrics: ')
    logger.info(f'R^2: {r2}')
    logger.info(f'RMSE: {rmse}')
    logger.info(f'MAE: {mae}')
    logger.info(f'Max Error Value: {max_val_error}')
    logger.info('-'*30)
    model.save(f'../model/{model_name}_{hidden_units}HU.h5')
    logger.info(f'Model saved at ../model/{model_name}_{hidden_units}HU.h5')

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