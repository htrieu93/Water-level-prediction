import pickle
import os
import logging.config
import tensorflow as tf
from tf.keras.models import load_model
from model import LSTM_model, GRU_model, BiLSTM_model
from src import config
from src.utils.metrics import calculate_loss
from src.utils.write_result import write_result
from src.utils.load_data_model import load_data_model
from src.utils.set_global_variables import scenario, n_steps, lead_time, target, model_name

# Model params
n_units = 150
dropout = .2
learning_rate = 1e-3
epochs = 200
batch_size = 256

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('loggers')

def train(trainX, trainY, testX, testY, model_name,
          pretrain=True, n_units=n_units, dropout=dropout,
          lr=learning_rate, epochs=epochs, batch_size=batch_size):

    if model_name == 'LSTM':
        model = LSTM_model(trainX, n_units)
    elif model_name == 'GRU':
        model = GRU_model(trainX, n_units)
    elif model_name == 'Bi-LSTM':
        model = BiLSTM_model(trainX, n_units)

    if pretrain:
        logger.info('Loading pretrain model...')
        model = tf.keras.models.load_model(f'../../model/{model_name}.h5')
    else:
        logger.info('Training model...')
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        history = model.fit(trainX, trainY,
                            epochs=epochs,
                            batch_size=batch_size, validation_data=(testX, testY), verbose=0,
                            shuffle=False,
                            callbacks=[callback])
    return model

if __name__ == '__main__':
    logger.info('Load data for modelling...')
    trainX, trainY, testX, testY, trueY = load_model_data()

    logger.info('Training model...')
    train(trainX, trainY, testX, testY, trueY,
          pretrain=args.pretrain,
          model_name=model_name,
          )