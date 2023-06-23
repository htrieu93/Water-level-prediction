import pandas as pd
import tensorflow as tf 
import random 
from .model import LSTM_model, GRU_model, BiLSTM_model

# Reproducibility
seed = 42
os.PYTHONHASHSEED = 0
tf.keras.utils.set_random_seed(seed)
tf.config.experimental.enable_op_determinism()
np.random.seed(seed)
tf.random.set_seed(seed)

# Model params
n_units = 100
dropout = .2
learning_rate = 1e-3
epochs = 200
forecast = False

def run_model(date_df, 
              trainX, 
              trainY, 
              testX, 
              testY, 
              list_n_step=[24],
              list_lead_time=[1, 6, 12], 
              station, 
              test_yr, 
              model_name):
    model_res = pd.DataFrame(['Model', 'Lag', 'Lead', 'R2', 'RMSE', 'MAE', 'MAX_VAL_ERROR'])
    for n_steps in list_n_step:
        for lead_time in list_lead_time:
            if forecast:
                dataset_excel = pd.DataFrame({'Date': date_df[lead_time:-1].reset_index(drop=True), 
                                              'True': testY.iloc[lead_time-1:, 0].reset_index(drop=True)})
            else:
                dataset_excel = pd.DataFrame({'Date': date_df[lead_time-1:].reset_index(drop=True), 
                                              'True': testY.iloc[lead_time-1:, 0].reset_index(drop=True)})
            for model_name in ['LSTM', 'GRU', 'Bi-LSTM']:
            # for model_name in ['LSTM']:
                print(f'Train {model_name} with {n_steps} lag and {lead_time} lead')
                trainX_rescale, testX_rescale, trainY_rescale, testY_rescale, testY, rescaler = preprocess_dataset_e(trainX, trainY, testX, testY, n_steps, lead_time)
                
                if model_name == 'LSTM':
                    model = LSTM_model(trainX_rescale)
                elif model_name == 'GRU':
                    model = GRU_model(trainX_rescale)
                elif model_name == 'Bi-LSTM':
                    model = BLSTM_model(trainX_rescale)
                elif model_name == 'Cust_BLSTM':
                    model = BLSTM_custom_model(trainX_rescale)
                
                model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
                history = model.fit(trainX_rescale, trainY_rescale, 
                                    epochs=epochs, 
                                    batch_size=256, validation_data=(testX_rescale, testY_rescale), verbose=0,
                                    shuffle=False,
                                    callbacks=[callback])
                
                predY = model.predict(testX_rescale)
                predY_rescale = rescaler.inverse_transform(predY)
                r2, rmse, mae, max_val_error = calculate_loss(testY[n_steps+lead_time:], predY_rescale)
                model_res = model_res.append({'Model': model_name,
                                              'Lag': n_steps,
                                              'Lead': lead_time,
                                              'R2':r2, 
                                              'RMSE':rmse, 
                                              'MAE':mae, 
                                              'MAX_VAL_ERROR':max_val_error}, ignore_index=True)
                dataset_excel = pd.concat([dataset_excel, 
                                           pd.DataFrame({f'{model_name}': np.append(np.array([0]*(n_steps+lead_time)), predY_rescale.ravel())})], axis=1)
                
            dataset_excel.to_excel(f'../res/{station}{test_yr}_{model_name}_{lead_time}h_lead_{n_steps}h_lag.xlsx')
    return(model_res)
