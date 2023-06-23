import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf 
import matplotlib.pyplot as plt
import random as python_random
import datetime, os
from sklearn.preprocessing import MinMaxScaler

def train_test_split(df, train_ratio):
    data_len = df.shape[0]
    train_len = int(data_len*train_ratio)
    trainX, testX = df.iloc[:train_len, 1:], df.iloc[train_len:, 1:]
    trainY, testY = df.iloc[:train_len, 0].values.reshape(-1, 1), df.iloc[train_len:, 0].values.reshape(-1, 1)
    return trainX, testX, trainY, testY

def split_sequence(X, y, n_steps, lead_time=1, use_forecast=False):
    seq_X, seq_Y = list(), list()
    for i in range(len(y)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if use_forecast and end_ix+lead_time+1 > len(y)-1:
            break
        elif end_ix+lead_time > len(y)-1:
            break

        # gather input and output parts of the pattern
        if use_forecast:  
            fcst_x = np.concatenate([np.zeros(shape=(n_steps, 3)), np.array(np.cumsum(X[end_ix:end_ix+lead_time, -3:], axis=0)[-1]).reshape((1, -1))], axis=0)
            seq_x, seq_y = np.concatenate([X[i:end_ix+1, :-3], fcst_x], axis=1), y[end_ix+lead_time, -1]
        else:  
            seq_x, seq_y = X[i:end_ix+1], y[end_ix+lead_time, -1]
        
        seq_X.append(seq_x)
        seq_Y.append(seq_y)
    return np.array(seq_X), np.array(seq_Y)

def min_max_scale(trainX, trainY, testX, testY):
    feat_rescaler = MinMaxScaler().fit(trainX)
    target_rescaler = MinMaxScaler().fit(trainY)
    
    trainX_rescale = feat_rescaler.transform(trainX) 
    testX_rescale = feat_rescaler.transform(testX) 
    trainY_rescale = target_rescaler.transform(trainY) 
    testY_rescale = target_rescaler.transform(testY) 
    return trainX_rescale, testX_rescale, trainY_rescale, testY_rescale, target_rescaler

def inverse_scale(predY, min_, scale_):
    predY_rescale = (predY - min_)/scale_
    return predY_rescale

def preprocess_data(trainX, testX, trainY, testY, n_steps, lead_time):
    # trainX, testX, trainY, testY = train_test_split(df, train_ratio)
    
    trainX_rescale, testX_rescale, trainY_rescale, testY_rescale, target_rescaler = min_max_scale(trainX, testX, trainY, testY)

    trainX_rescale, trainY_rescale = split_sequence(trainX_rescale, trainY_rescale, n_steps, lead_time, use_forecast=True)
    testX_rescale, testY_rescale = split_sequence(testX_rescale, testY_rescale, n_steps, lead_time, use_forecast=True)
    
    trainX_rescale = trainX_rescale.reshape((trainX_rescale.shape[0], trainX_rescale.shape[2], trainX_rescale.shape[1]))
    testX_rescale = testX_rescale.reshape((testX_rescale.shape[0], testX_rescale.shape[2], testX_rescale.shape[1]))
    return trainX_rescale, testX_rescale, trainY_rescale, testY_rescale, testY, target_rescaler

def write_result(model_name, date_df, dataset_df, train_ratio, predY, lead_time, n_steps, forecast, scenario):
    if forecast:
        dataset_excel = pd.DataFrame({'Date': date_df[int(date_df.shape[0]*train_ratio)+lead_time:-1].reset_index(drop=True), 
                                      'True': dataset_df.iloc[int(dataset_df.shape[0]*train_ratio)+lead_time-1:, 0].reset_index(drop=True), 
                                      f'{model_name}': np.append(np.array([0]*(n_steps+1)), predY.ravel())})
    else:
        dataset_excel = pd.DataFrame({'Date': date_df[int(date_df.shape[0]*train_ratio)+lead_time-1:].reset_index(drop=True), 
                                      'True': dataset_df.iloc[int(dataset_df.shape[0]*train_ratio)+lead_time-1:, 0].reset_index(drop=True), 
                                      f'{model_name}': np.append(np.array([0]*(n_steps+1)), predY.ravel())})
    dataset_excel.to_excel(f'/content/drive/My Drive/Water level prediction paper/Prediction/{model_name}_SC{scenario}_{lead_time}h_lead_{n_steps}h_lag.xlsx')
