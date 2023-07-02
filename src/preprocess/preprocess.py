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

parser.add_argument('-s', '--scenario')
parser.add_argument('-m', '--model')

args = parser.parse_args()

DATA_2010_PATH = '../data/XQHh_Nhatle_2010.xlsx'
DATA_2012_PATH = '../data/XQHh_Nhatle_2012.xlsx'
DATA_2016_PATH = '../data/XQHh_Nhatle_2016.xlsx'
DATA_2020_PATH = '../data/XQHh_Nhatle_2020.xlsx'

def main():
    logger.info('Start loading data...')
    df_2010, df_2012, df_2016, df_2020 = load_data(DATA_2010_PATH, DATA_2012_PATH, DATA_2016_PATH, DATA_2020_PATH)
    logger.info('Finished loading data...')
    logger.info(f'{df_2010.shape}')
    logger.info(f'{df_2012.shape}')
    logger.info(f'{df_2016.shape}')
    logger.info(f'{df_2020.shape}')

    logger.info('Preprocessing data...')
    if args.scenario == 1:
        df_2010, df_2012, df_2016, df_2020, date = preprocess_data(df_2010, df_2012, df_2016, df_2020)
    elif args.scenario in [2,3]:
        df_2012, df_2020, date = preprocess_data(df_2010, df_2012, df_2016, df_2020)

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
    logger.info(f'DATASET SHAPE: {dataset.shape}')


if __name__ == '__main__':
    main()