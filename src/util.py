import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf 
import matplotlib.pyplot as plt
import random as python_random
import datetime, os
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import os, datetime

def load_data(DATA_2010_PATH, DATA_2012_PATH, DATA_2016_PATH, DATA_2020_PATH):
  # Read 2010 data
  MN_data_2010 = pd.read_excel(DATA_2010_PATH, 
                               sheet_name='QHh_2010', skiprows=2, 
                               names=['Time', 'Q_KienGiang', 'H_KienGiang', 'H_LeThuy', 'H_DongHoi'])
  LM_data_2010 = pd.read_excel(DATA_2010_PATH, 
                               sheet_name='Xh_Oct2Dec_2010', skiprows=2, 
                               names=['Time', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])
  
  # Read 2012 data
  MN_data_2012 = pd.read_excel(DATA_2012_PATH, 
                               sheet_name='QHh_2012', skiprows=2, 
                               names=['Time', 'Q_KienGiang', 'H_KienGiang', 'H_LeThuy', 'H_DongHoi'])
  LM_data_2012 = pd.read_excel(DATA_2012_PATH, 
                               sheet_name='Xh_2012', skiprows=2, usecols='A,F,G,H',
                               names=['Time', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])
  
  # Read 2016 data
  MN_data_2016_1 = pd.read_excel(DATA_2016_PATH, 
                                 sheet_name='H_KG_DH', skiprows=2, 
                                 names=['Time', 'H_KienGiang', 'H_DongHoi'])
  MN_data_2016_2 = pd.read_excel(DATA_2016_PATH, 
                                 sheet_name='Hh_LeThuy_Oct', skiprows=2, usecols='A,F',
                                 names=['Time', 'H_LeThuy'])
  LM_data_2016 = pd.read_excel(DATA_2016_PATH, 
                               sheet_name='Xh_Donghoi_Sep2Nov', skiprows=2, usecols='G,H,I',
                               names=['Date', 'Time', 'LM_DongHoi'])
  
  # Read 2020 data
  MN_data_2020 = pd.read_excel(DATA_2020_PATH, 
                               sheet_name='QHh_2020', skiprows=2, 
                               names=['Time', 'Q_KienGiang', 'H_KienGiang', 'H_LeThuy', 'H_DongHoi'])
  LM_data_2020 = pd.read_excel(DATA_2020_PATH, 
                               sheet_name='Xh_2020', skiprows=2,  
                               names=['Time', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])

  # Concatenate 2 parts of MN of 2016
  MN_data_2016 = MN_data_2016_1.merge(MN_data_2016_2, on='Time')
  
  # Concat water level and rainfall features (only for 2012, 2020 since 2010, 2016 don't have rainfall data)
  df_2012 = MN_data_2012.merge(LM_data_2012, on='Time', how='left')
  df_2020 = MN_data_2020.merge(LM_data_2020, on='Time', how='left')
    
  return MN_data_2010, df_2012, MN_data_2016, df_2020
  
def preprocess_data(df_2010, df_2012, df_2016, df_2020):
  # Simulate rainfall of 02/2012 using rainfall of 02/2020
  LM_data_2012_Feb = df_2020.loc[(df_2020['Time']>='2/1/2020') & (df_2020['Time']<'3/1/2020')]
  LM_data_2012_Feb['Time'] = LM_data_2012_Feb['Time'].apply(lambda x: x.replace(year=2012))
  df_2012 = pd.concat([df_2012, LM_data_2012_Feb]).sort_values('Time')
  
  # Convert water level of 2012 from cm to m
  for col in ['H_KienGiang', 'H_LeThuy', 'H_DongHoi']:
    df_2012[col] = df_2012[col] / 100
  
  # Convert Time column to datetime
  for df in [df_2010, df_2012, df_2016, df_2020]:
    df['Time'] = pd.to_datetime(df['Time'])
  
  date_a = pd.concat([df_2010['Time'], df_2012['Time'], df_2016['Time'], df_2020['Time']])
  date_b = pd.concat([df_2012['Time'], df_2020['Time']])
  
  # Fill NaN rainfall 2020 has 2020-12-31 22:59:59.990 instead of 2020-12-31 23:00:00
  df_2020.fillna(0, inplace=True)
  return df_2010, df_2012, df_2016, df_2020, date_a, date_b

def feature_engineering(df_2012, df_2020):
  # Create t+1 rainfall feature
  df_2012['LM_LeThuy_lead1'] =  df_2012['LM_LeThuy'].shift(periods=-1)
  df_2012['LM_KienGiang_lead1'] =  df_2012['LM_KienGiang'].shift(periods=-1)
  df_2012['LM_DongHoi_lead1'] =  df_2012['LM_DongHoi'].shift(periods=-1)
  df_2020['LM_LeThuy_lead1'] =  df_2020['LM_LeThuy'].shift(periods=-1)
  df_2020['LM_KienGiang_lead1'] =  df_2020['LM_KienGiang'].shift(periods=-1)
  df_2020['LM_DongHoi_lead1'] =  df_2020['LM_DongHoi'].shift(periods=-1)
  return df_2012, df_2020

def create_data_scenario(data_2010, data_2012, data_2016, data_2020, scenario=1):
  if scenario == 1:
    # Dataset consisting of 3 stations MN (2010, 2012, 2016, 2020)
    data_2010_a = data_2010[['H_LeThuy', 'H_KienGiang', 'H_DongHoi', 'H_LeThuy']].astype(np.float32)
    data_2012_a = data_2012[['H_LeThuy', 'H_KienGiang', 'H_DongHoi', 'H_LeThuy']].astype(np.float32)
    data_2016_a = data_2016[['H_LeThuy', 'H_KienGiang', 'H_DongHoi', 'H_LeThuy']].astype(np.float32)
    data_2020_a = data_2020[['H_LeThuy', 'H_KienGiang', 'H_DongHoi', 'H_LeThuy']].astype(np.float32)
    return pd.concat([data_2010_a, data_2012_a, data_2016_a, data_2020_a])
  elif scenario == 2:
    # Dataset consisting of 3 stations MN  and LM (2012, 2020)
    data_2012_b = data_2012[['H_LeThuy', 'H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi']].astype(np.float32)
    data_2020_b = data_2020[['H_LeThuy', 'H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi']].astype(np.float32)
    return pd.concat([data_2012_b, data_2020_b])
  elif scenario == 3:
    # Dataset consisting of 3 stations MN  and LM and predicted LM (2012, 2020)
    data_2012_c = data_2012[['H_LeThuy', 'H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi',
                             'LM_LeThuy_lead1', 'LM_KienGiang_lead1', 'LM_DongHoi_lead1']].astype(np.float32)
    data_2020_c = data_2020[['H_LeThuy', 'H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi',
                             'LM_LeThuy_lead1', 'LM_KienGiang_lead1', 'LM_DongHoi_lead1']].astype(np.float32)
    return pd.concat([data_2012_c, data_2020_c])

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
