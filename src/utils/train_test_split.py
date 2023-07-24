import pandas as pd
import numpy as np

def train_test_split(df, train_ratio, target_col):
    data_len = df.shape[0]
    train_len = int(data_len*train_ratio)
    trainX, testX = df.iloc[:train_len], df.iloc[train_len:]
    trainY, testY = df.iloc[:train_len][target_col].values.reshape(-1, 1), df.iloc[train_len:][target_col].values.reshape(-1, 1)
    return trainX, testX, trainY, testY
