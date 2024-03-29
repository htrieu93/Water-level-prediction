import pickle
import logging.config
import pandas as pd
from src import config
from src.utils import *
from src.utils.set_global_variables import scenario, n_steps, lead_time, target

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('loggers')

DATA_2010_PATH = r'XQHh_Nhatle_2010.xlsx'
DATA_2012_PATH = r'XQHh_Nhatle_2012.xlsx'
DATA_2016_PATH = r'XQHh_Nhatle_2016.xlsx'
DATA_2020_PATH = r'XQHh_Nhatle_2020.xlsx'

if __name__ == '__main__':
    logger.info('Loading data...')
    MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016, MN_data_2020, LM_data_2020 = load_data(DATA_2010_PATH,
                                                                                                   DATA_2012_PATH,
                                                                                                   DATA_2016_PATH,
                                                                                                   DATA_2020_PATH)

    logger.info('Cleaning data...')
    df_2010, df_2012, df_2016, df_2020, date = clean_data(MN_data_2010, MN_data_2012, LM_data_2012,
                                                          MN_data_2016, MN_data_2020, LM_data_2020,
                                                          scenario)

    logger.info('Creating data scenario...')
    if args.scenario == 1:
        dataset = create_scenario(df_2010, df_2012, df_2016, df_2020,
                                  scenario=scenario,
                                  feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy'])
    elif args.scenario == 2:
        dataset = create_scenario(df_2012, df_2020,
                                  scenario=scenario,
                                  feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy',
                                            'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])
    elif args.scenario == 3:
        # Create artificial forecast rainfall values for 2012 and 2020
        df_2012, df_2020 = feature_engineering(df_2012, df_2020)
        dataset = create_scenario(df_2012, df_2020,
                                  scenario=scenario,
                                  feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy',
                                            'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi',
                                            'LM_LeThuy_lead1', 'LM_KienGiang_lead1', 'LM_DongHoi_lead1'])
    logger.info(f'DATASET SHAPE: {dataset.shape}')

    # Visualization
    plot_pacf_tar(dataset, target)  # Plot Partial Autocorrelation Function chart
    plot_avg_water_level(dataset, date, target)  # Plot Average Water Level of Target

    # Generating inputs and output for modelling
    trainX, testX, trainY, testY = train_test_split(dataset, train_ratio=.8, target_col=target)

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
    trainX_rescale, trainY_rescale = split_data_by_year(trainX, trainY, n_steps, lead_time, scenario, train_year_len)

    # Splitting sequence for testing set
    year_idx = 0
    test_year_len = [year_idx]
    for year in pd.DatetimeIndex(date).year.unique():
        year_idx = date.loc[pd.DatetimeIndex(date).year == year].shape[0] + year_idx
        if year_idx - trainX.shape[0] > 0:
            test_year_len.append(year_idx - trainX.shape[0])
    testX_rescale, testY_rescale = split_data_by_year(testX, testY, n_steps, lead_time, scenario, test_year_len)

    trainX_rescale = trainX_rescale.reshape((trainX_rescale.shape[0], trainX_rescale.shape[2], trainX_rescale.shape[1]))
    testX_rescale = testX_rescale.reshape((testX_rescale.shape[0], testX_rescale.shape[2], testX_rescale.shape[1]))

    logger.info('Saving training data...')
    pickle.dump(trainX_rescale,
                open(f'../postprocess/x_train_rescale_s{scenario}_{target}_{n_steps}_lag_{lead_time}_lead.pkl',
                     'wb'))
    pickle.dump(trainY_rescale,
                open(f'../postprocess/y_train_rescale_s{scenario}_{target}_{n_steps}_lag_{lead_time}_lead.pkl',
                     'wb'))
    pickle.dump(testX_rescale,
                open(f'../postprocess/x_test_rescale_s{scenario}_{target}_{n_steps}_lag_{lead_time}_lead.pkl',
                     'wb'))
    pickle.dump(testY_rescale,
                open(f'../postprocess/y_test_rescale_s{scenario}_{target}_{n_steps}_lag_{lead_time}_lead.pkl',
                     'wb'))
    pickle.dump(testY,
                open(f'../postprocess/y_test_s{scenario}_{n_steps}_{target}_{n_steps}_lag_{lead_time}_lead.pkl',
                     'wb'))
