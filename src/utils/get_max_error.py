import pandas as pd
import numpy as np

def get_max_error(y_test, y_pred):
    return abs(y_test - y_pred).max().values[0]