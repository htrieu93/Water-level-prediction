import logging, argparse
import logging.config
from util import *
from config import *

# Run once at startup:
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('loggers')

parser = argparse.ArgumentParser(
                    prog='WaterLevelPrediction',
                    description='Preprocessing data for the water prediction paper')

parser.add_argument('-s', '--scenario', type=int)
parser.add_argument('-y', '--target')
parser.add_argument('-n', '--n_steps')
parser.add_argument('-l', '--lead_time')
parser.add_argument('-p', '--pretrain')
parser.add_argument('-m', '--model')

args = parser.parse_args()

DATA_2010_PATH = r'XQHh_Nhatle_2010.xlsx'
DATA_2012_PATH = r'XQHh_Nhatle_2012.xlsx'
DATA_2016_PATH = r'XQHh_Nhatle_2016.xlsx'
DATA_2020_PATH = r'XQHh_Nhatle_2020.xlsx'

def main():
    logger.info('Start loading data...')
    MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016, MN_data_2020, LM_data_2020 = load_data(DATA_2010_PATH, DATA_2012_PATH, DATA_2016_PATH, DATA_2020_PATH)
    logger.info('Finished loading data...')

    logger.info('Start cleaning data...')
    if args.scenario == 1:
        df_2010, df_2012, df_2016, df_2020, date = clean_data(MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016,
                                                              MN_data_2020, LM_data_2020, args.scenario)
        logger.info(f'{df_2010.shape}')
        logger.info(f'{df_2012.shape}')
        logger.info(f'{df_2016.shape}')
        logger.info(f'{df_2020.shape}')

    elif args.scenario in [2,3]:
        df_2012, df_2020, date = clean_data(df_2010, df_2012, df_2016, df_2020, args.scenario)
        logger.info(f'{df_2012.shape}')
        logger.info(f'{df_2020.shape}')
    logger.info('Finished cleaning data...')

    logger.info('Start creating data scenario...')
    # Create leading rainfall features for Scenario 3
    if args.scenario == 3:
      df_2012, df_2020 = feature_engineering(df_2012, df_2020)

    if args.scenario == 1:
      dataset = create_data_scenario(df_2010, df_2012, df_2016, df_2020,
                                     scenario=args.scenario,
                                     feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy'])
    elif args.scenario == 2:
      dataset = create_data_scenario(df_2012, df_2020,
                                     scenario=args.scenario,
                                     feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])
    elif args.scenario == 3:
      dataset = create_data_scenario(df_2012, df_2020,
                                     scenario=args.scenario,
                                     feat_col=['H_KienGiang', 'H_DongHoi', 'H_LeThuy', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi',
                                               'LM_LeThuy_lead1', 'LM_KienGiang_lead1', 'LM_DongHoi_lead1'])
    logger.info('Finished creating data scenario...')
    logger.info(f'DATASET SHAPE: {dataset.shape}')

    trainX_rescale, testX_rescale, trainY_rescale, testY_rescale, testY, target_rescaler = preprocess_data(dataset, date, args.target, args.n_steps,
                                                                                                           args.lead_time, train_ratio=0.8)

if __name__ == '__main__':
    main()