import pandas as pd
import numpy as np
from src.utils.split_sequence import split_sequence

def split_data_by_year(X, Y, lag_time, lead_time, scenario, year_len):
    X_lst = []
    Y_lst = []
    for i in range(len(year_len) - 1):
        split_X, split_Y = split_sequence(X[year_len[i]:year_len[i + 1]],
                                          Y[year_len[i]:year_len[i + 1]],
                                          lag_time, lead_time, scenario=scenario)
        X_lst.append(split_X)
        Y_lst.append(split_Y)

    X_res = np.concatenate(X_lst)
    Y_res = np.concatenate(Y_lst)
    return X_res, Y_res