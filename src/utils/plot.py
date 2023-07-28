import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf

def plot_pacf(df, lags=20):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.rcParams['font.size'] = '22'
    plot_pacf(df.iloc[:,0], lags=lags, zero=False, alpha=0.05, ax=ax)
    ax.set_xticks(range(0,20,2))
    ax.set_xlabel('Lag(s)')
    ax.set_ylabel('Partial Autocorrelation')
    ax.set_title(None)
    plt.savefig('../../pacf_plot.png')

def plot_avg_water_level(df, months, avg_water_lvl):
    first_warning_level = 1.2
    second_warning_level = 2.2
    third_warning_level = 2.7
    x_1 = np.linspace(0,12,100)
    y_1 = 1.2

    fig, ax = plt.subplots(figsize=(22, 10))
    plt.rcParams['font.size'] = '20'
    sns.boxplot(x=months, y=avg_water_lvl, ax=ax)
    ax.set_xticklabels(['January', 'February', 'March', 'April', 'May', 'June', 'July',
                        'August', 'September', 'October', 'November', 'December'], rotation=45)
    ax.set_xlabel(None)
    ax.set_ylabel('Le Thuy water level (m)')
    plt.axhline(y = third_warning_level, color = 'r', label='Alarm 3 (H = 2.7m)')
    plt.axhline(y = second_warning_level, color = 'r', linestyle = '--', label='Alarm 2 (H = 2.2m)')
    plt.axhline(y = first_warning_level, color = 'r', linestyle = ':', label='Alarm 1 (H = 1.2m)')
    plt.legend(loc='upper left')
    plt.savefig('../../yearly_avg_water_level.png')

def plot_compare_params(R2, RMSE, MAE, MEV, param,
                        xlabel,
                        ylabel):
    metrics = ['R^2', 'RMSE', 'MAE', 'MEV']
    xlabel = [[1, 2, 3]] * 4
    ylabel = [[50, 100, 150]] * 4

    plt.figure(figsize=(10, 10), dpi=80)
    for j, (metric, label, xlabel, ylabel) in enumerate(zip([R2, RMSE, MAE, MEV], metrics, xlabel, ylabel)):
        ax = plt.subplot(2, 2, j + 1)
        ax.plot(ylabel, metric, marker='o', mfc='r')
        ax.set_xticks(range(50, 200, 50))
        ax.set_xticklabels(ylabel)
        ax.set_ylabel(label)
        ax.set_xlabel(f'Number of {param}')
    plt.show()

def plot_model_loss(train_loss, val_loss):
    plt.figure(figsize=(15, 8))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('train loss vs. val loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_compare_result(df, LSTM_pred, GRU_pred, BiLSTM_pred, y_true, n_steps, lead_time, train_ratio):
    date_range_test = df[int(df.shape[0]*train_ratio) + n_steps + lead_time:].dt.strftime('%m/%d/%Y')
    date_range_test.reset_index(inplace=True, drop=True)
    xticks = np.arange(0, BiLSTM_predY_rescale.shape[0], 500)
    xticks_label = date_range_test.iloc[xticks]

    plt.figure(figsize=(23, 10))
    plt.plot(LSTM_pred, color='red', label='LSTM')
    plt.plot(GRU_pred, color='blue', label='GRU')
    plt.plot(BiLSTM_pred, color='green', label='Bi-LSTM')
    plt.plot(y_true[n_steps+lead_time:], color='black', label='True')
    plt.xticks(xticks, xticks_label,
               rotation=29)
    plt.title(f'Lead time T+{lead_time}')
    plt.xlabel('Time')
    plt.ylabel('Water level(m)')
    plt.legend()
    plt.savefig(f'T+{lead_time}.jpg', bbox_inches='tight')
