import logging
import argparse
import sys
import pickle
import logging.config
import pandas as pd
from src import config
from src.utils import *

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('loggers')

parser = argparse.ArgumentParser(
                    prog='WaterLevelPrediction',
                    description='Preprocessing data for the water prediction paper')

parser.add_argument('-s', '--scenario', type=int)
parser.add_argument('-y', '--target')
parser.add_argument('-n', '--n_steps', type=int)
parser.add_argument('-l', '--lead_time', type=int)

args = parser.parse_args()

DATA_2010_PATH = r'XQHh_Nhatle_2010.xlsx'
DATA_2012_PATH = r'XQHh_Nhatle_2012.xlsx'
DATA_2016_PATH = r'XQHh_Nhatle_2016.xlsx'
DATA_2020_PATH = r'XQHh_Nhatle_2020.xlsx'

def main():
    logger.info('Loading data...')
    MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016, MN_data_2020, LM_data_2020 = load_data(DATA_2010_PATH, DATA_2012_PATH, DATA_2016_PATH, DATA_2020_PATH)

    logger.info('Cleaning data...')
    df_2010, df_2012, df_2016, df_2020, date = clean_data(MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016,
                                                          MN_data_2020, LM_data_2020, args.scenario)

    logger.info('Creating data scenario...')
    if args.scenario == 1:
      dataset = create_scenario(df_2010, df_2012, df_2016, df_2020,
                                     scenario=args.scenario,
                                     feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy'])
    elif args.scenario == 2:
      dataset = create_scenario(df_2012, df_2020,
                                     scenario=args.scenario,
                                     feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])
    elif args.scenario == 3:
      # Create artificial forecast rainfall values for 2012 and 2020
      df_2012, df_2020 = feature_engineering(df_2012, df_2020)
      dataset = create_scenario(df_2012, df_2020,
                                     scenario=args.scenario,
                                     feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi',
                                               'LM_LeThuy_lead1', 'LM_KienGiang_lead1', 'LM_DongHoi_lead1'])
    logger.info(f'DATASET SHAPE: {dataset.shape}')

    trainX, testX, trainY, testY = train_test_split(dataset, train_ratio=.8, target_col=args.target)

    # Rescale data by min-max
    trainX, testX, trainY, testY = min_max_scale(trainX, trainY, testX, testY)

    # Splitting sequence for training set by year
    year_idx = 0
    train_year_len = [year_idx]
    for year in pd.DatetimeIndex(date).year.unique():
        year_idx = date.loc[pd.DatetimeIndex(date).year == year].shape[0] + year_idx
        if year_idx <= trainX.shape[0]:
            train_year_len.append(year_idx)
        else:
            train_year_len.append(trainX.shape[0])
    trainX_rescale, trainY_rescale = split_data_by_year(trainX, trainY, args.n_steps, args.lead_time, args.scenario, train_year_len)

    # Splitting sequence for testing set
    year_idx = 0
    test_year_len = [year_idx]
    for year in pd.DatetimeIndex(date).year.unique():
        year_idx = date.loc[pd.DatetimeIndex(date).year == year].shape[0] + year_idx
        if year_idx - trainX.shape[0] > 0:
            test_year_len.append(year_idx - trainX.shape[0])
    testX_rescale, testY_rescale = split_data_by_year(testX, testY, args.n_steps, args.lead_time, args.scenario, test_year_len)

    trainX_rescale = trainX_rescale.reshape((trainX_rescale.shape[0], trainX_rescale.shape[2], trainX_rescale.shape[1]))
    testX_rescale = testX_rescale.reshape((testX_rescale.shape[0], testX_rescale.shape[2], testX_rescale.shape[1]))

    logger.info('Saving training data...')
    pickle.dump(trainX_rescale, open('../postprocess/x_train_rescale.pkl', 'wb'))
    pickle.dump(trainY_rescale, open('../postprocess/y_train_rescale.pkl', 'wb'))
    pickle.dump(testX_rescale, open('../postprocess/x_test_rescale.pkl', 'wb'))
    pickle.dump(testY_rescale, open('../postprocess/y_test_rescale.pkl', 'wb'))
    pickle.dump(testY, open('../postprocess/y_test.pkl', 'wb'))

if __name__ == '__main__':
    main()