import pandas as pd
import numpy as np
import pandas as pd
import os
import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None

def feature_engineering(df_2012, df_2020):
    # Create t+1 rainfall feature
    df_2012['LM_LeThuy_lead1'] =  df_2012['LM_LeThuy'].shift(periods=-1)
    df_2012['LM_KienGiang_lead1'] =  df_2012['LM_KienGiang'].shift(periods=-1)
    df_2012['LM_DongHoi_lead1'] =  df_2012['LM_DongHoi'].shift(periods=-1)
    df_2020['LM_LeThuy_lead1'] =  df_2020['LM_LeThuy'].shift(periods=-1)
    df_2020['LM_KienGiang_lead1'] =  df_2020['LM_KienGiang'].shift(periods=-1)
    df_2020['LM_DongHoi_lead1'] =  df_2020['LM_DongHoi'].shift(periods=-1)
    return df_2012, df_2020