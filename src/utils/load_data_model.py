import os
import pickle

def load_data_model():
    os.chdir(os.getcwd() + r'/data/postprocess')

    trainX = pickle.load(open(f'x_train_rescale_s{scenario}_{target}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
    trainY = pickle.load(open(f'y_train_rescale_s{scenario}_{target}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
    testX = pickle.load(open(f'x_test_rescale_s{scenario}_{target}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
    testY = pickle.load(open(f'y_test_rescale_s{scenario}_{target}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
    trueY = pickle.load(open(f'y_test_s{scenario}_{target}_{n_steps}_lag_{lead_time}_lead.pkl', 'rb'))
    return trainX, trainY, testX, testY, trueY