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

# plan for the class
# accepted inputs: stock_name, start_date, end_date
# dataframes should be stored in a dictionary
# three methods: ExponentialSmoothing, VAR, and auto arima (pmdarima/pm)
# HOW TO BUILD THE WALK-FORWARD VALIDATION SO IT WORKS FOR ALL METHODS?

# plotting functions outside of class


def prepare_data(
        df,
):
    train = df.iloc[:-N_TEST]
    test = df.iloc[-N_TEST:]
    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    return train, test, train_idx, test_idx


def plot_fitted_forecast(
        df,
        train_idx,
        test_idx,
        model_result,
        col=None,
        forecast_df=None,
):
    if forecast_df is None:
        df.loc[train_idx, 'fitted'] = model_result.fittedvalues
        df.loc[test_idx, 'forecast'] = np.array(model_result.forecast(N_TEST))
    else:
        col = "Scaled_" + col
        df.loc[train_idx, 'fitted'] = model_result.fittedvalues[col]
        df.loc[test_idx, 'forecast'] = forecast_df[col].values

    df[[f"{col}", 'fitted', 'forecast']][-108:].plot(figsize=(15, 5))
    plt.legend()
    plt.show()
#
# last_10_values = df_UAL['ETSfitted'].tail(N_test)
# df_UAL.loc[last_10_values.index, 'ETSfitted'].plot(style='r', label='Prediction')
# plt.legend()


class StocksForecast:
    def __init__(
            self,
            stock_name_list: list[str] = ('UAL', 'WMT', 'PFE'),
            start_date: str = '2018-01-01',
            end_date: str = '2022-12-31',
    ):
        self.dfs = dict()
        for name in stock_name_list:
            self.dfs[name] = yf.download(name, start=start_date, end=end_date)
            self.dfs[name]['Diff'] = self.dfs[name]['Close'].diff(1)
            self.dfs[name]['Log'] = np.log(self.dfs[name]['Close'])
            # self.dfs[name]['Close_lag1'] = self.dfs[name]['Close'].shift(1)

    def run_ets(
            self,
            stock_name='UAL',
            col='Close',
    ):
        train, test, train_idx, test_idx = prepare_data(self.dfs[stock_name])

        model = ExponentialSmoothing(train[col].dropna(),
                                     trend='add',
                                     seasonal='add',
                                     seasonal_periods=252,)
        result = model.fit()

        plot_fitted_forecast(self.dfs[stock_name], train_idx, test_idx, result, col)

    def run_var(
            self,
            stock_list=('UAL', 'WMT', 'PFE'),
            col='Close',
    ):

        df_all = pd.DataFrame(index=self.dfs[stock_list[0]].index)
        for stock in stock_list:
            df_all = df_all.join(self.dfs[stock][col].dropna())
            df_all.rename(columns={col: f"{stock}_{col}"}, inplace=True)

        train, test, train_idx, test_idx = prepare_data(df_all)

        stock_cols = df_all.columns.values

        for value in stock_cols:
            scaler = StandardScaler()
            train[f'Scaled_{value}'] = scaler.fit_transform(train[[value]])
            test[f'Scaled_{value}'] = scaler.transform(test[[value]])
            df_all.loc[train_idx, f'Scaled_{value}'] = train[f'Scaled_{value}']
            df_all.loc[test_idx, f'Scaled_{value}'] = test[f'Scaled_{value}']

        cols = ['Scaled_' + value for value in stock_cols]

        # plot_acf(train['Scaled_ual'])
        # plot_pacf(train['Scaled_ual'])
        # plot_pacf(train['Scaled_pfe'])

        model = VAR(train[cols])
        result = model.fit(maxlags=40, method='mle', ic='aic')
        lag_order = result.k_ar

        prior = train.iloc[-lag_order:][cols].to_numpy()
        forecast_df = pd.DataFrame(result.forecast(prior, N_TEST), columns=cols)

        plot_fitted_forecast(df_all, train_idx, test_idx, result, stock_cols[0], forecast_df)
        #
        # train_pred = df_all.loc[train_idx, 'var_pred_ual'].iloc[lag_order:]
        # train_true = df_all.loc[train_idx, 'Scaled_ual'].iloc[lag_order:]
        #
        # print("Train R2: ", r2_score(train_true, train_pred))
        #
        # test_pred = df_all.loc[test_idx, 'var_forecast_ual']
        # test_true = df_all.loc[test_idx, 'Scaled_ual']
        # print("Test R2:", r2_score(test_true, test_pred))

    def run_arima(
            self
    ):
        pass

# # Auto ARIMA
#
# arima_model = pm.auto_arima(train['ual'],
#                             trace=True,
#                             suppress_warning=True,
#                             seasonal=True,
#                             m=12)
#
# arima_model.summary()
#
# test_pred, confint = arima_model.predict(n_periods=Ntest, return_conf_int=True)
#
# fig, ax = plt.subplots(figsize=(15, 5))
# ax.plot(test.index, test['ual'], label='data')
# ax.plot(test.index, test_pred, label='forecast')
# ax.fill_between(test.index, confint[:,0], confint[:,1],
#                 color='red', alpha=.3)
# ax.legend()
#
# train_pred = arima_model.predict_in_sample(end=-1)
#
# fig, ax = plt.subplots(figsize=(15, 5))
# ax.plot(df_all.index, df_all['ual'], label='data')
# ax.plot(train.index[12:], train_pred[12:], label='fitted')
# ax.plot(test.index, test_pred, label='forecast')
# ax.fill_between(test.index, confint[:,0], confint[:,1],
#                 color='red', alpha=.3)
# ax.legend()





# # Walk-forward Validation (Lazy Programmer)
#
# h = 20 # 4 weeks
# steps = 10
# Ntest = len(df_UAL) - h - steps + 1
#
# # Hyperparameters to try
# trend_type_list = ['add', 'mul']
# seasonal_type_list = ['add', 'mul']
# init_method_list = ['estimated', 'heuristic', 'legacy-heristic']
# use_boxcox_list = [True, False, 0]
#
# def walkforward(
#     df,
#     col,
#     trend_type,
#     seasonal_type,
#     debug=False,
# ):
#   # store errors
#   errors = []
#   seen_last = False
#   steps_completed = 0
#
#   for end_of_train in range(Ntest, len(df) - h + 1):
#     train = df.iloc[:end_of_train]
#     test = df.iloc[end_of_train:end_of_train + h]
#
#     if test.index[-1] == df.index[-1]:
#       seen_last = True
#
#     steps_completed += 1
#
#     hw = ExponentialSmoothing(
#         train[col],
#         trend=trend_type,
#         seasonal=seasonal_type,
#         seasonal_periods=40,
#     )
#
#     result_hw = hw.fit()
#
#     forecast = result_hw.forecast(h)
#     error = mean_squared_error(test[col], np.array(forecast))
#     errors.append(error)
#
#   if debug:
#       print("seen_last:", seen_last)
#       print("steps completed:", steps_completed)
#
#   return np.mean(errors)
#
# tuple_of_option_lists = (trend_type_list, seasonal_type_list,)
#
# best_score = float('inf')
# best_options = None
#
# for x in itertools.product(*tuple_of_option_lists):
#   score = walkforward(df_UAL, "Close", *x)
#
#   if score < best_score:
#     print("Best score so far:", score)
#
#     best_score = score
#     best_options = x
#
# trend_type, seasonal_type = best_options
# print(trend_type)
# print(seasonal_type)
#
#


if __name__ == "__main__":
    N_TEST = 20

    ts = StocksForecast()

    ts.run_ets('UAL', col='Log')
    ts.run_var(col='Log')
