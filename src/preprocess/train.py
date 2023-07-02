import pandas as pd
import tensorflow as tf 
from .model import LSTM_model, GRU_model, BiLSTM_model

import pickle

# Model params
n_units = 100
dropout = .2
learning_rate = 1e-3
epochs = 200
forecast = False

logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger('loggers')

def train_model(trainX, testX, trainY, testY, model_name):

    if model_name == 'LSTM':
        model = LSTM_model(trainX)
    elif model_name == 'GRU':
        model = GRU_model(trainX)
    elif model_name == 'Bi-LSTM':
        model = BiLSTM_model(trainX)

    logger.info('Training model...')
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(trainX_rescale, trainY_rescale,
                        epochs=epochs,
                        batch_size=256, validation_data=(testX, testY), verbose=0,
                        shuffle=False,
                        callbacks=[callback])

    predY = model.predict(testX_rescale)
    r2, rmse, mae, max_val_error = calculate_loss(testY_rescale[n_steps+lead_time:], predY)
    logger.info('Finished.')
    logger.info('Metrics: ')
    logger.info(f'R^2: {r2}')
    logger.info(f'RMSE: {rmse}')
    logger.info(f'MAE: {mae}')
    logger.info(f'Max Error Value: {max_val_error}')
    logger.info('-'*30)
    model.save(f'./model/{model_name}_{hidden_units}HU.h5')
    logger.info(f'Model saved at ./model/{model_name}_{hidden_units}HU.h5')

def predict(model_weights_path):
    # Loads the weights
    model.load_weights(checkpoint_path)

if __name__ == '__main__':
