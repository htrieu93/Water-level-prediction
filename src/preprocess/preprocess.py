import logging
import argparse
import logging.config
from util import *
from config import *

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('loggers')

parser = argparse.ArgumentParser(
                    prog='WaterLevelPrediction',
                    description='Preprocessing data for the water prediction paper')

parser.add_argument('-s', '--scenario', type=int)
parser.add_argument('-y', '--target')
parser.add_argument('-n', '--n_steps')
parser.add_argument('-l', '--lead_time')

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
      dataset = create_data_scenario(df_2010, df_2012, df_2016, df_2020,
                                     scenario=args.scenario,
                                     feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy'])
    elif args.scenario == 2:
      dataset = create_data_scenario(df_2012, df_2020,
                                     scenario=args.scenario,
                                     feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])
    elif args.scenario == 3:
      # Create artificial forecast rainfall values for 2012 and 2020
      df_2012, df_2020 = feature_engineering(df_2012, df_2020)
      dataset = create_data_scenario(df_2012, df_2020,
                                     scenario=args.scenario,
                                     feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi',
                                               'LM_LeThuy_lead1', 'LM_KienGiang_lead1', 'LM_DongHoi_lead1'])
    logger.info(f'DATASET SHAPE: {dataset.shape}')

    preprocess_data(dataset, date, args.target, int(args.n_steps), int(args.lead_time), train_ratio=0.8, scenario=args.scenario)

if __name__ == '__main__':
    main()