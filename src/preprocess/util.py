import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd

import os, datetime, pickle

from sklearn.preprocessing import MinMaxScaler

def load_data(DATA_2010_PATH, DATA_2012_PATH, DATA_2016_PATH, DATA_2020_PATH):
    os.chdir(os.getcwd() + "/data/")
    # Read 2010 data
    MN_data_2010 = pd.read_excel(DATA_2010_PATH,
                               sheet_name='QHh_2010', skiprows=2,
                               names=['Time', 'Q_KienGiang', 'H_KienGiang', 'H_LeThuy', 'H_DongHoi'])

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

    # Read 2020 data
    MN_data_2020 = pd.read_excel(DATA_2020_PATH,
                               sheet_name='QHh_2020', skiprows=2,
                               names=['Time', 'Q_KienGiang', 'H_KienGiang', 'H_LeThuy', 'H_DongHoi'])
    LM_data_2020 = pd.read_excel(DATA_2020_PATH,
                               sheet_name='Xh_2020', skiprows=2,
                               names=['Time', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])

    # Concatenate 2 parts of MN of 2016
    MN_data_2016 = MN_data_2016_1.merge(MN_data_2016_2, on='Time')
    return MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016, MN_data_2020, LM_data_2020
  
def preprocess_data(MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016, MN_data_2020, LM_data_2020, scenario):
    # Convert Time column to datetime
    for df in [MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016, MN_data_2020, LM_data_2020]:
        df['Time'] = pd.to_datetime(df['Time'])

    # Concat water level and rainfall features (only for 2012, 2020 since 2010, 2016 don't have rainfall data)
    df_2012 = MN_data_2012.merge(LM_data_2012, on='Time', how='left')
    df_2020 = MN_data_2020.merge(LM_data_2020, on='Time', how='left')

    # Simulate rainfall of 02/2012 using rainfall of 02/2020
    LM_data_2012_Feb = df_2020.loc[(df_2020['Time']>='2/1/2020') &
                                   (df_2020['Time']<'3/1/2020')]
    LM_data_2012_Feb['Time'] = LM_data_2012_Feb['Time'].apply(lambda x: x.replace(year=2012))
    df_2012 = pd.concat([df_2012, LM_data_2012_Feb]).sort_values('Time')

    # Convert water level of 2012 from cm to m
    for col in ['H_KienGiang', 'H_LeThuy', 'H_DongHoi']:
        df_2012[col] = df_2012[col] / 100

    # Fill NaN rainfall 2020 has 2020-12-31 22:59:59.990 instead of 2020-12-31 23:00:00
    df_2020.fillna(0, inplace=True)

    if scenario == 1:
        date = pd.concat([df_2010['Time'], df_2012['Time'], df_2016['Time'], df_2020['Time']])
        return df_2010, df_2012, df_2016, df_2020, date
    elif scenario in [2, 3]:
        date = pd.concat([df_2012['Time'], df_2020['Time']])
        return df_2012, df_2020, date

def feature_engineering(df_2012, df_2020):
    # Create t+1 rainfall feature
    df_2012['LM_LeThuy_lead1'] =  df_2012['LM_LeThuy'].shift(periods=-1)
    df_2012['LM_KienGiang_lead1'] =  df_2012['LM_KienGiang'].shift(periods=-1)
    df_2012['LM_DongHoi_lead1'] =  df_2012['LM_DongHoi'].shift(periods=-1)
    df_2020['LM_LeThuy_lead1'] =  df_2020['LM_LeThuy'].shift(periods=-1)
    df_2020['LM_KienGiang_lead1'] =  df_2020['LM_KienGiang'].shift(periods=-1)
    df_2020['LM_DongHoi_lead1'] =  df_2020['LM_DongHoi'].shift(periods=-1)
    return df_2012, df_2020

def create_data_scenario(data_2010, data_2012, data_2016, data_2020, scenario, feat_col):
    if scenario == 1:
        # Dataset consisting of 3 stations MN (2010, 2012, 2016, 2020)
        res_df = pd.concat([data_2010, data_2012, data_2016, data_2020]).astype(np.float32)
    elif scenario in [2, 3]:
        # Dataset consisting of 3 stations MN  and LM (2012, 2020)
        res_df = pd.concat([data_2012, data_2020]).astype(np.float32)
    return res_df[feat_col]

def train_test_split(df, train_ratio, target_col):
    data_len = df.shape[0]
    train_len = int(data_len*train_ratio)
    trainX, testX = df.iloc[:train_len], df.iloc[train_len:]
    trainY, testY = df.iloc[:train_len, target_col].values.reshape(-1, 1), df.iloc[train_len:, target_col].values.reshape(-1, 1)
    return trainX, testX, trainY, testY

def split_sequence(X, y, n_steps, lead_time=1, scenario=1):
    seqX, seqY = list(), list()
    for i in range(len(y)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if use_forecast and end_ix+lead_time+1 > len(y)-1:
            break
        elif end_ix+lead_time > len(y)-1:
            break

        # gather input and output parts of the pattern
        if scenario == 3:  
            fcst_x = np.concatenate([np.zeros(shape=(n_steps, 3)), np.array(np.cumsum(X[end_ix:end_ix+lead_time, -3:], axis=0)[-1]).reshape((1, -1))], axis=0)
            seqx, seqy = np.concatenate([X[i:end_ix+1, :-3], fcst_x], axis=1), y[end_ix+lead_time, -1]
        else:  
            seqx, seqy = X[i:end_ix+1], y[end_ix+lead_time, -1]
        
        seqX.append(seqx)
        seqY.append(seqy)
    return np.array(seqX), np.array(seqY)

def min_max_scale(trainX, trainY, testX, testY):
    feat_rescaler = MinMaxScaler().fit(trainX)
    target_rescaler = MinMaxScaler().fit(trainY)
    pickle.dump(target_rescaler, open('./pkld/target_rescale', 'wb'))

    trainX_rescale = feat_rescaler.transform(trainX) 
    testX_rescale = feat_rescaler.transform(testX) 
    trainY_rescale = target_rescaler.transform(trainY) 
    testY_rescale = target_rescaler.transform(testY)
    return trainX_rescale, testX_rescale, trainY_rescale, testY_rescale

def preprocess_data(trainX, testX, trainY, testY, date, n_steps, lead_time, scenario):
    trainX_rescale, testX_rescale, trainY_rescale, testY_rescale = min_max_scale(trainX, testX, trainY, testY)

    year_len = []
    year_idx = 0
    for year in pd.DatetimeIndex(date).year.unique():
        year_idx =  date.loc[pd.DatetimeIndex(date).year == year].shape[0] + year_idx
        year_len.append(year_idx)

    trainX_rescale_lst = []
    trainY_rescale_lst = []
    for i in range(len(year_idx) - 1):
        trainX_rescale, trainY_rescale = split_sequence(trainX_rescale[year_idx[i]:year_idx[i+1]],
                                                        trainY_rescale[year_idx[i]:year_idx[i+1]], n_steps, lead_time)
        trainX_rescale_lst.append(trainX_rescale)
        trainY_rescale_lst.append(trainY_rescale)

    trainX_rescale = np.concatenate(trainX_rescale_lst)
    trainY_rescale = np.concatenate(trainY_rescale_lst)
    testX_rescale, testY_rescale = split_sequence(testX_rescale, testY_rescale, n_steps, lead_time)

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
    dataset_excel.to_excel(f'../pred/{model_name}_SC{scenario}_{lead_time}h_lead_{n_steps}h_lag.xlsx')
