import pandas as pd

def load_csv(dest_file, dtype):
    os.chdir(os.getcwd() + "/data/postprocess")
    with open(dest_file, 'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter=',',
                               quotechar='"')
        data = [data for data in data_iter]
    return np.asarray(data, dtype=dtype)

def write_result(model_name, date_df, dataset_df, train_ratio, predY, lead_time, n_steps, forecast, scenario):
    if forecast:
        dataset_excel = pd.DataFrame({'Date': date_df[int(date_df.shape[0]*train_ratio)+lead_time:-1].reset_index(drop=True),
                                      'True': dataset_df.iloc[int(dataset_df.shape[0]*train_ratio)+lead_time-1:, 0].reset_index(drop=True),
                                      f'{model_name}': np.append(np.array([0]*(n_steps+1)), predY.ravel())})
    else:
        dataset_excel = pd.DataFrame({'Date': date_df[int(date_df.shape[0]*train_ratio)+lead_time-1:].reset_index(drop=True),
                                      'True': dataset_df.iloc[int(dataset_df.shape[0]*train_ratio)+lead_time-1:, 0].reset_index(drop=True),
                                      f'{model_name}': np.append(np.array([0]*(n_steps+1)), predY.ravel())})
    dataset_excel.to_excel(f'../pred/{model_name}_SC{scenario}_{lead_time}h_lead_{n_steps}h_lag.xlsx')