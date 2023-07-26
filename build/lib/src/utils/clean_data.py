import pandas as pd
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

def clean_data(MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016, MN_data_2020, LM_data_2020, scenario):
    # Convert Time column to datetime
    for df in [MN_data_2010, MN_data_2012, LM_data_2012, MN_data_2016, MN_data_2020, LM_data_2020]:
        df['Time'] = pd.to_datetime(df['Time'])

    # Concat water level and rainfall features (only for 2012, 2020 since 2010, 2016 don't have rainfall data)
    df_2012 = MN_data_2012.merge(LM_data_2012, on='Time', how='left')
    df_2020 = MN_data_2020.merge(LM_data_2020, on='Time', how='left')

    # Simulate rainfall of 02/2012 using rainfall of 02/2020
    LM_data_2012_Feb = df_2020.loc[(df_2020['Time']>='2/1/2020') &
                                   (df_2020['Time']<'3/1/2020')]
    LM_data_2012_Feb['Time'] = LM_data_2012_Feb['Time'].apply(lambda x: x.replace(year=2012))
    df_2012 = pd.concat([df_2012, LM_data_2012_Feb]).sort_values('Time')

    # Convert water level of 2012 from cm to m
    for col in ['H_KienGiang', 'H_LeThuy', 'H_DongHoi']:
        df_2012[col] = df_2012[col] / 100

    # Fill NaN rainfall 2020 has 2020-12-31 22:59:59.990 instead of 2020-12-31 23:00:00
    df_2020.fillna(0, inplace=True)

    if scenario == 1:
        date = pd.concat([MN_data_2010['Time'], df_2012['Time'], MN_data_2016['Time'], df_2020['Time']])
        MN_data_2010.drop(columns=['Time'], inplace=True)
        df_2012.drop(columns=['Time'], inplace=True)
        MN_data_2016.drop(columns=['Time'], inplace=True)
        df_2020.drop(columns=['Time'], inplace=True)

        return MN_data_2010, df_2012, MN_data_2016, df_2020, date
    elif scenario in [2, 3]:
        date = pd.concat([df_2012['Time'], df_2020['Time']])
        df_2012.drop(columns=['Time'], inplace=True)
        df_2020.drop(columns=['Time'], inplace=True)

        return None, df_2012, None, df_2020, date