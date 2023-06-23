from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

def plot_pacf(df):
  fig, ax = plt.subplots(figsize=(12, 12))
  plt.rcParams['font.size'] = '22'
  plot_pacf(df.iloc[:,0], lags=20, zero=False, alpha=0.05, ax=ax)
  ax.set_xticks(range(0,20,2))
  ax.set_xlabel('Lag(s)')
  ax.set_ylabel('Partial Autocorrelation')
  ax.set_title(None)
  plt.show()

def plot_avg_water_level(months, avg_water_lvl):
  fig, ax = plt.subplots(figsize=(22, 10))
  plt.rcParams['font.size'] = '20'
  x_1 = np.linspace(0,12,100)
  y_1 = 1.2
  sns.boxplot(x=months, y=avg_water_lvl, ax=ax)
  ax.set_xticklabels(['January', 'February', 'March', 'April', 'May', 'June', 'July',
                      'August', 'September', 'October', 'November', 'December'], rotation=45)
  ax.set_xlabel(None)
  ax.set_ylabel('Le Thuy water level (m)')
  plt.axhline(y = 2.7, color = 'r', label='Alarm 3 (H = 2.7m)')
  plt.axhline(y = 2.2, color = 'r', linestyle = '--', label='Alarm 2 (H = 2.2m)')
  plt.axhline(y = 1.2, color = 'r', linestyle = ':', label='Alarm 1 (H = 1.2m)')
  plt.legend(loc='upper left')
  plt.show()

def plot_compare_params(R2, RMSE, MAE, MEV):
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
      ax.set_xlabel('Number of hidden units')
  
  plt.show()
