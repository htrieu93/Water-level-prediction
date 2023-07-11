import logging, argparse
from util import *
from config import *

FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger('loggers')

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-s', '--scenario', type=int)
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

    logger.info('Start reprocessing data...')
    if args.scenario == 1:
        df_2010, df_2012, df_2016, df_2020, date = preprocess_data(MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016, MN_data_2020, LM_data_2020)
        logger.info(f'{df_2010.shape}')
        logger.info(f'{df_2012.shape}')
        logger.info(f'{df_2016.shape}')
        logger.info(f'{df_2020.shape}')

    elif args.scenario in [2,3]:
        df_2012, df_2020, date = preprocess_data(df_2010, df_2012, df_2016, df_2020)
        logger.info(f'{df_2012.shape}')
        logger.info(f'{df_2020.shape}')
    logger.info('Finished preprocessing data...')

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

    train_test_split(dataset, train_ratio, target_col)

if __name__ == '__main__':
    main()