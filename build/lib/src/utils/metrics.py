import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, max_error, mean_squared_error

def calculate_loss(y_test, y_pred):
    # calculate metrics (Max value error, RMSE, R^2, MAE)
    max_val_error = np.round(abs(y_test - y_pred).max(), 3)
    rmse = np.round(float(mean_squared_error(y_test, y_pred, squared=False)), 3)
    r2 = np.round(float(r2_score(y_test, y_pred)), 3)
    mae = np.round(float(mean_absolute_error(y_test, y_pred)), 3)
    return r2, rmse, mae, max_val_error
