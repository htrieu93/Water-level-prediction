import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

def split_sequence(X, y, lag_time, lead_time=1, scenario=1):
    seqX, seqY = list(), list()
    for i in range(len(y)):
    # find the end of this pattern
        end_ix = i + lag_time
        # check if we are beyond the sequence
        if scenario == 3 and end_ix+lead_time+1 > len(y)-1:
            break
        elif end_ix+lead_time > len(y)-1:
            break

        # gather input and output parts of the pattern
        if scenario == 3:  
            fcst_x = np.concatenate([np.zeros(shape=(lag_time, 3)), np.array(np.cumsum(X[end_ix:end_ix+lead_time, -3:], axis=0)[-1]).reshape((1, -1))], axis=0)
            seqx, seqy = np.concatenate([X[i:end_ix+1, :-3], fcst_x], axis=1), y[end_ix+lead_time, -1]
        else:  
            seqx, seqy = X[i:end_ix+1], y[end_ix+lead_time, -1]
        
        seqX.append(seqx)
        seqY.append(seqy)
    return np.array(seqX), np.array(seqY)