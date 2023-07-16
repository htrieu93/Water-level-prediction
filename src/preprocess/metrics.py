import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, max_error, mean_squared_error

def get_max_error(y_test, y_pred):
    return abs(y_test - y_pred).max().values[0]

def calculate_loss(y_test, y_pred):
    max_val_error = get_max_error(y_test, y_pred)

    # calculate LOSS
    rmse = np.round_(float(mean_squared_error(y_test, y_pred, squared=False)), 3)
    r2 = np.round_(float(r2_score(y_test, y_pred)), 3)
    mae = np.round_(float(mean_absolute_error(y_test, y_pred)), 3)
    return r2, rmse, mae, max_val_error
