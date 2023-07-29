import tensorflow as tf
import pickle
import os
import argparse
import logging.config
from tf.keras.models import load_model
from model import LSTM_model, GRU_model, BiLSTM_model
from src.utils.metrics import calculate_loss
from src.utils.write_result import write_result
from src import config
from src.utils.set_global_variables import scenario, n_steps, lead_time, target

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
                    description='Training and predicting with RNN models')

parser.add_argument('-p', '--pretrain', action='store_true')
parser.add_argument('-m', '--model')

args = parser.parse_args()

def load_model_data():
    os.chdir(os.getcwd() + r'/data/postprocess')
    try:
        trainX = pickle.load(open(f'x_train_rescale_s{scenario}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
        trainY = pickle.load(open(f'y_train_rescale_s{scenario}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
        testX = pickle.load(open(f'x_test_rescale_s{scenario}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
        testY = pickle.load(open(f'y_test_rescale_s{scenario}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
        trueY = pickle.load(open(f'y_test_s{scenario}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
        return trainX, trainY, testX, testY, trueY.
xcccccccccccccxxxxxxxxxxxxxxxxxxx
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

def predict(testX, trueY,
            pretrain=False, model_path=None, model_name=None):
    # Loads pretrained model
    if pretrain:
        model = load_model(model_path)

    # Make prediction and evaluate metrics
    predY = model.predict(testX)
    r2, rmse, mae, max_val_error = calculate_loss(trueY[args.n_steps + args.lead_time:],
                                                  predY)
    logger.info('Metrics: ')
    logger.info(f'R^2: {r2}')
    logger.info(f'RMSE: {rmse}')
    logger.info(f'MAE: {mae}')
    logger.info(f'Max Error Value: {max_val_error}')
    logger.info('-' * 30)
    model.save(f'../../model/{model_name}_{n_steps}_lag_{lead_time}_lead.h5')
    logger.info(f'Model saved at ../../model/{model_name}_{n_steps}_lag_{lead_time}_lead.h5')

if __name__ == '__main__':
    logger.info('Load data for modelling...')
    trainX, trainY, testX, testY, trueY = load_model_data()

    logger.info('Training model...')
    train(trainX, trainY, testX, testY, trueY,
          pretrain=args.pretrain,
          model_name=args.model,
          )

    logger.info('Generating prediction and evaluating model...')
    if args.pretrain:
        predict(testX, trueY,
                pretrain=args.pretrain,
                model_name=args.model)
    else:
        predict(testX, trueY)