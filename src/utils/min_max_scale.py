import pandas as pd
import numpy as np
import pandas as pd
import os
import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None

def min_max_scale(trainX, trainY, testX, testY):
    feat_rescaler = MinMaxScaler().fit(trainX)
    target_rescaler = MinMaxScaler().fit(trainY)
    pickle.dump(target_rescaler, open(r'../pickled/target_rescaler.pkl', 'wb'))

    trainX_rescale = feat_rescaler.transform(trainX)
    testX_rescale = feat_rescaler.transform(testX)
    trainY_rescale = target_rescaler.transform(trainY)
    testY_rescale = target_rescaler.transform(testY)
    return trainX_rescale, testX_rescale, trainY_rescale, testY_rescale