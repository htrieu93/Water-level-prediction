import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

def create_scenario(data_2010, data_2012, data_2016, data_2020, scenario, feat_col):
    if scenario == 1:
        # Dataset consisting of 3 stations MN (2010, 2012, 2016, 2020)
        res_df = pd.concat([data_2010, data_2012, data_2016, data_2020]).astype(np.float32)
    elif scenario in [2, 3]:
        # Dataset consisting of 3 stations MN  and LM (2012, 2020)
        res_df = pd.concat([data_2012, data_2020]).astype(np.float32)
    return res_df[feat_col]