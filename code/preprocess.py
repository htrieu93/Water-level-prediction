import pandas as pd
import os, datetime

def load_data(DATA_2010_PATH, DATA_2012_PATH, DATA_2016_PATH, DATA_2020_PATH):
  # Read 2010 data
  MN_data_2010 = pd.read_excel('../data/XQHh_Nhatle_2010.xlsx', 
                               sheet_name='QHh_2010', skiprows=2, 
                               names=['Time', 'Q_KienGiang', 'H_KienGiang', 'H_LeThuy', 'H_DongHoi'])
  LM_data_2010 = pd.read_excel('../data/XQHh_Nhatle_2010.xlsx', 
                               sheet_name='Xh_Oct2Dec_2010', skiprows=2, 
                               names=['Time', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])
  
  # Read 2012 data
  MN_data_2012 = pd.read_excel('../data/XQHh_Nhatle_2012.xlsx', 
                               sheet_name='QHh_2012', skiprows=2, 
                               names=['Time', 'Q_KienGiang', 'H_KienGiang', 'H_LeThuy', 'H_DongHoi'])
  LM_data_2012 = pd.read_excel('../data/XQHh_Nhatle_2012.xlsx', 
                               sheet_name='Xh_2012', skiprows=2, usecols='A,F,G,H',
                               names=['Time', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])
  
  # Read 2016 data
  MN_data_2016_1 = pd.read_excel('../data/)XHh_Nhatle_2016.xlsx', 
                                 sheet_name='H_KG_DH', skiprows=2, 
                                 names=['Time', 'H_KienGiang', 'H_DongHoi'])
  MN_data_2016_2 = pd.read_excel('../data/)XHh_Nhatle_2016.xlsx', 
                                 sheet_name='Hh_LeThuy_Oct', skiprows=2, usecols='A,F',
                                 names=['Time', 'H_LeThuy'])
  LM_data_2016 = pd.read_excel('../data/)XHh_Nhatle_2016.xlsx', 
                               sheet_name='Xh_Donghoi_Sep2Nov', skiprows=2, usecols='G,H,I',
                               names=['Date', 'Time', 'LM_DongHoi'])
  
  # Read 2020 data
  MN_data_2020 = pd.read_excel('../data/XQHh_Nhatle_2020.xlsx', 
                               sheet_name='QHh_2020', skiprows=2, 
                               names=['Time', 'Q_KienGiang', 'H_KienGiang', 'H_LeThuy', 'H_DongHoi'])
  LM_data_2020 = pd.read_excel('../data/XQHh_Nhatle_2020.xlsx', 
                               sheet_name='Xh_2020', skiprows=2,  
                               names=['Time', 'LM_KienGiang', 'LM_LeThuy', 'LM_DongHoi'])

def preprocess_data():
  # Simulate LM of 02/2012 using LM of 02/2020
  LM_data_2012_Feb = LM_data_2020.loc[(LM_data_2020['Time'] >= '2/1/2020')&(LM_data_2020['Time'] < '3/1/2020')]
  LM_data_2012_Feb['Time'] = LM_data_2012_Feb['Time'].apply(lambda x: x.replace(year=2012))
  LM_data_2012 = pd.concat([LM_data_2012, LM_data_2012_Feb]).sort_values('Time')
  
  # Convert MN of 2012 from cm to m
  MN_data_2012['H_KienGiang'] = MN_data_2012['H_KienGiang'] / 100
  MN_data_2012['H_LeThuy'] = MN_data_2012['H_LeThuy'] / 100
  MN_data_2012['H_DongHoi'] = MN_data_2012['H_DongHoi'] / 100
  
  # Concatenate 2 parts of MN of 2016
  MN_data_2016 = MN_data_2016_1.merge(MN_data_2016_2, on='Time')
  
  # Convert Time column to datetime
  MN_data_2010['Time'] = pd.to_datetime(MN_data_2010['Time'])
  MN_data_2012['Time'] = pd.to_datetime(MN_data_2012['Time'])
  MN_data_2016['Time'] = pd.to_datetime(MN_data_2016['Time'])
  MN_data_2020['Time'] = pd.to_datetime(MN_data_2020['Time'])
  
  date_a = pd.concat([MN_data_2010['Time'], MN_data_2012['Time'], MN_data_2016['Time'], MN_data_2020['Time']])
  date_b = pd.concat([MN_data_2012['Time'], MN_data_2020['Time']])
  
  # Concat MN and LM features
  data_2010 = MN_data_2010
  data_2012 = MN_data_2012.merge(LM_data_2012, on='Time', how='left')
  data_2016 = MN_data_2016
  data_2020 = MN_data_2020.merge(LM_data_2020, on='Time', how='left')
  
  # Fill NaN LM 2020 has 2020-12-31 22:59:59.990 instead of 2020-12-31 23:00:00
  data_2020.fillna(0, inplace=True)
  return data_2010, data_2012, data_2016, data_2020, date_a, date_b

def add_lead_features(data_2012, data_2020):
  # Create t+1 rainfall feature
  data_2012['LM_LeThuy_lead1'] =  data_2012['LM_LeThuy'].shift(periods=-1)
  data_2012['LM_KienGiang_lead1'] =  data_2012['LM_KienGiang'].shift(periods=-1)
  data_2012['LM_DongHoi_lead1'] =  data_2012['LM_DongHoi'].shift(periods=-1)
  data_2020['LM_LeThuy_lead1'] =  data_2020['LM_LeThuy'].shift(periods=-1)
  data_2020['LM_KienGiang_lead1'] =  data_2020['LM_KienGiang'].shift(periods=-1)
  data_2020['LM_DongHoi_lead1'] =  data_2020['LM_DongHoi'].shift(periods=-1)
  return data_2012, data_2020

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
