import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
import pmdarima as pm

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

df_UAL = yf.download('UAL', start="2018-01-01", end="2022-12-31") # United Airlines
df_WMT = yf.download('WMT', start="2018-01-01", end="2022-12-31") # Walmart
df_PFE = yf.download('PFE', start="2018-01-01", end="2022-12-31") # Pfizer

df_UAL.head()

print(df_UAL.index.freq)

df_UAL['Close'].plot(figsize=(15, 5))

df_UAL['Return'] = np.log(df_PFE['Close'].pct_change(1) + 1) # log return
df_UAL['Return'].plot(figsize=(15, 5))

df_UAL['Close_lag1'] = df_UAL['Close'].shift(1)

N_test = 40
train = df_UAL.iloc[:-N_test]
test = df_UAL.iloc[-N_test]
ets_model = ExponentialSmoothing(train['Close'].dropna(),
                                 trend='add',
                                 seasonal='add',
                                 seasonal_periods=252,)
ets_result = ets_model.fit()

train_idx = df_UAL.index <= train.index[-1]
test_idx = df_UAL.index > train.index[-1]

df_UAL.loc[train_idx, 'ETSfitted'] = ets_result.fittedvalues
df_UAL.loc[test_idx, 'ETSfitted'] = np.array(ets_result.forecast(N_test))
df_UAL[['Close_lag1', 'ETSfitted']][-108:].plot(figsize=(15, 5))

last_10_values = df_UAL['ETSfitted'].tail(N_test)
df_UAL.loc[last_10_values.index, 'ETSfitted'].plot(style='r', label='Prediction')
plt.legend()

# Walk-forward Validation (Lazy Programmer)

h = 20 # 4 weeks
steps = 10
Ntest = len(df_UAL) - h - steps + 1

# Hyperparameters to try
trend_type_list = ['add', 'mul']
seasonal_type_list = ['add', 'mul']
init_method_list = ['estimated', 'heuristic', 'legacy-heristic']
use_boxcox_list = [True, False, 0]

def walkforward(
    df,
    col,
    trend_type,
    seasonal_type,
    debug=False,
):
  # store errors
  errors = []
  seen_last = False
  steps_completed = 0

  for end_of_train in range(Ntest, len(df) - h + 1):
    train = df.iloc[:end_of_train]
    test = df.iloc[end_of_train:end_of_train + h]

    if test.index[-1] == df.index[-1]:
      seen_last = True

    steps_completed += 1

    hw = ExponentialSmoothing(
        train[col],
        trend=trend_type,
        seasonal=seasonal_type,
        seasonal_periods=40,
    )

    result_hw = hw.fit()

    forecast = result_hw.forecast(h)
    error = mean_squared_error(test[col], np.array(forecast))
    errors.append(error)

  if debug:
      print("seen_last:", seen_last)
      print("steps completed:", steps_completed)

  return np.mean(errors)

tuple_of_option_lists = (trend_type_list, seasonal_type_list,)

best_score = float('inf')
best_options = None

for x in itertools.product(*tuple_of_option_lists):
  score = walkforward(df_UAL, "Close", *x)

  if score < best_score:
    print("Best score so far:", score)

    best_score = score
    best_options = x

trend_type, seasonal_type = best_options
print(trend_type)
print(seasonal_type)

# The following implements VAR with the closing prices of the three stocks
# Use "Return" may be better since they are at the same scale

ual = df_UAL[['Close']].copy().dropna().rename(columns={'Close':'ual'})
wmt = df_WMT[['Close']].copy().dropna().rename(columns={'Close':'wmt'})
pfe = df_PFE[['Close']].copy().dropna().rename(columns={'Close':'pfe'})

df_all = ual.merge(wmt.merge(pfe, how='inner', on='Date'), how='inner', on='Date')

df_all.head()

df_all.plot(figsize=(15, 5))

Ntest = 40
train = df_all.iloc[:-Ntest].copy()
test = df_all.iloc[-Ntest:].copy()

train_idx = df_all.index <= train.index[-1]
test_idx = df_all.index > train.index[-1]

stock_cols = df_all.columns.values

for value in stock_cols:
  scaler = StandardScaler()
  train[f'Scaled_{value}'] = scaler.fit_transform(train[[value]])
  test[f'Scaled_{value}'] = scaler.transform(test[[value]])
  df_all.loc[train_idx, f'Scaled_{value}'] = train[f'Scaled_{value}']
  df_all.loc[test_idx, f'Scaled_{value}'] = test[f'Scaled_{value}']

cols = modified_list = ['Scaled_' + value for value in stock_cols]

train[cols].plot(figsize=(15, 5))

plot_acf(train['Scaled_ual'])

plot_pacf(train['Scaled_ual'])

plot_pacf(train['Scaled_pfe'])

var_model = VAR(train[cols])
lag_order_results = var_model.select_order(maxlags=20)
print(lag_order_results.selected_orders)

var_result = var_model.fit(maxlags=20, ic='aic')

lag_order = var_result.k_ar
prior = train.iloc[-lag_order:][cols].to_numpy()
var_forecast = var_result.forecast(prior, Ntest)

var_forecast_df = pd.DataFrame(var_forecast, columns=cols)

df_all.loc[train_idx, 'var_pred_ual'] = var_result.fittedvalues['Scaled_ual']
df_all.loc[test_idx, 'var_forecast_ual'] = var_forecast_df['Scaled_ual'].values

df_all.filter(like='_ual').iloc[-100:].plot(figsize=(15, 5))

train_pred = df_all.loc[train_idx, 'var_pred_ual'].iloc[lag_order:]
train_true = df_all.loc[train_idx, 'Scaled_ual'].iloc[lag_order:]

print("Train R2: ", r2_score(train_true, train_pred))

test_pred = df_all.loc[test_idx, 'var_forecast_ual']
test_true = df_all.loc[test_idx, 'Scaled_ual']
print("Test R2:", r2_score(test_true, test_pred))

# Auto ARIMA

arima_model = pm.auto_arima(train['ual'],
                            trace=True,
                            suppress_warning=True,
                            seasonal=True,
                            m=12)

arima_model.summary()

test_pred, confint = arima_model.predict(n_periods=Ntest, return_conf_int=True)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(test.index, test['ual'], label='data')
ax.plot(test.index, test_pred, label='forecast')
ax.fill_between(test.index, confint[:,0], confint[:,1],
                color='red', alpha=.3)
ax.legend()

train_pred = arima_model.predict_in_sample(end=-1)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df_all.index, df_all['ual'], label='data')
ax.plot(train.index[12:], train_pred[12:], label='fitted')
ax.plot(test.index, test_pred, label='forecast')
ax.fill_between(test.index, confint[:,0], confint[:,1],
                color='red', alpha=.3)
ax.legend()

if __name__ == "__main__":
    pass
