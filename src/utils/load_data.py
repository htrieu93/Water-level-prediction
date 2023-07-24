import pandas as pd
import numpy as np
import os

pd.options.mode.chained_assignment = None

def load_data(DATA_2010_PATH, DATA_2012_PATH, DATA_2016_PATH, DATA_2020_PATH):
    os.chdir(os.getcwd() + "/data/raw")
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